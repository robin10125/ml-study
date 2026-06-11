// flash_fwd.cu — FlashAttention-2-style forward pass, teaching version.
// One thread per query row, fp32, no tensor cores. Verified against a CPU reference.
//
// Build:  nvcc -O3 -arch=native -o flash_fwd flash_fwd.cu
// Run:    ./flash_fwd
//
// Layout: Q,K,V,O are [B, H, N, d] contiguous ("BHND"); L is [B, H, N].

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// The kernel. Grid: (ceil(N/Br), H, B). Block: Br threads (one per query row).
// ---------------------------------------------------------------------------
template <int d, int Br, int Bc, bool causal>
__global__ void flash_fwd(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O,
                          float* __restrict__ L,
                          int N, float scale) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x * Br + tid;          // my global query index

    // Step 0: point at this (batch, head)'s slice.
    const size_t bh = (size_t)blockIdx.z * gridDim.y + blockIdx.y;
    const float* Qh = Q + bh * N * d;
    const float* Kh = K + bh * N * d;
    const float* Vh = V + bh * N * d;
    float*       Oh = O + bh * N * d;
    float*       Lh = L + bh * N;

    // K/V tiles are shared by the whole block: 2 * Bc * d * 4 bytes = 32 KB.
    __shared__ float Ks[Bc][d];
    __shared__ float Vs[Bc][d];

    // Per-row state lives in registers (q, o, s may partially spill; fine here).
    float q[d], o[d], s[Bc];
    float m = -INFINITY, l = 0.f;

    if (row < N)
        for (int k = 0; k < d; k++) q[k] = Qh[(size_t)row * d + k];
    for (int k = 0; k < d; k++) o[k] = 0.f;

    // Causal: tiles strictly above the diagonal are skipped entirely.
    const int j_end = causal ? min(N, blockIdx.x * Br + Br) : N;

    for (int j0 = 0; j0 < j_end; j0 += Bc) {
        // ---- load: all Br threads cooperatively stage K_j, V_j (coalesced) ----
        for (int idx = tid; idx < Bc * d; idx += Br) {
            int r = idx / d, c = idx % d;
            bool in_range = (j0 + r) < N;
            Ks[r][c] = in_range ? Kh[(size_t)(j0 + r) * d + c] : 0.f;
            Vs[r][c] = in_range ? Vh[(size_t)(j0 + r) * d + c] : 0.f;
        }
        __syncthreads();

        if (row < N) {
            // ---- scores: s = scale * q . K_j^T, with masking via -inf ----
            float mt = -INFINITY;
            for (int c = 0; c < Bc; c++) {
                float dot = 0.f;
#pragma unroll
                for (int k = 0; k < d; k++) dot += q[k] * Ks[c][k];
                bool masked = (j0 + c >= N) || (causal && j0 + c > row);
                s[c] = masked ? -INFINITY : dot * scale;
                mt = fmaxf(mt, s[c]);
            }

            // ---- online softmax update (Ch. 3/4): rescale history, fold tile ----
            float m_new = fmaxf(m, mt);
            float alpha = __expf(m - m_new);        // first tile: exp(-inf)=0
            l *= alpha;
#pragma unroll
            for (int k = 0; k < d; k++) o[k] *= alpha;
            for (int c = 0; c < Bc; c++) {
                float p = __expf(s[c] - m_new);     // masked: exp(-inf)=0
                l += p;
#pragma unroll
                for (int k = 0; k < d; k++) o[k] += p * Vs[c][k];
            }
            m = m_new;
        }
        __syncthreads();   // don't let fast threads overwrite Ks/Vs early
    }

    // ---- epilogue: the single deferred normalization, then one HBM write ----
    if (row < N) {
        float inv = 1.f / l;
        for (int k = 0; k < d; k++) Oh[(size_t)row * d + k] = o[k] * inv;
        Lh[row] = m + logf(l);
    }
}

// ---------------------------------------------------------------------------
// CPU reference: textbook 3-pass attention, fp32.
// ---------------------------------------------------------------------------
void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, float* L,
                   int B, int H, int N, int d, float scale, bool causal) {
    std::vector<float> s(N);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            size_t bh = ((size_t)b * H + h);
            const float* Qh = Q + bh * N * d;
            const float* Kh = K + bh * N * d;
            const float* Vh = V + bh * N * d;
            float*       Oh = O + bh * N * d;
            float*       Lh = L + bh * N;
            for (int i = 0; i < N; i++) {
                float m = -INFINITY;
                for (int j = 0; j < N; j++) {
                    float dot = 0.f;
                    for (int k = 0; k < d; k++) dot += Qh[i*d+k] * Kh[j*d+k];
                    s[j] = (causal && j > i) ? -INFINITY : dot * scale;
                    m = fmaxf(m, s[j]);
                }
                float l = 0.f;
                for (int j = 0; j < N; j++) { s[j] = expf(s[j] - m); l += s[j]; }
                for (int k = 0; k < d; k++) {
                    float acc = 0.f;
                    for (int j = 0; j < N; j++) acc += s[j] * Vh[j*d+k];
                    Oh[i*d+k] = acc / l;
                }
                Lh[i] = m + logf(l);
            }
        }
}

// ---------------------------------------------------------------------------
// Test harness.
// ---------------------------------------------------------------------------
constexpr int D = 64, BR = 64, BC = 64;

template <bool causal>
bool run_test(int B, int H, int N) {
    size_t n_qkv = (size_t)B * H * N * D, n_l = (size_t)B * H * N;
    std::vector<float> hQ(n_qkv), hK(n_qkv), hV(n_qkv);
    std::vector<float> hO(n_qkv), hL(n_l), refO(n_qkv), refL(n_l);
    srand(42);
    for (auto* v : {&hQ, &hK, &hV})
        for (auto& x : *v) x = 2.f * rand() / RAND_MAX - 1.f;

    float *dQ, *dK, *dV, *dO, *dL;
    CUDA_CHECK(cudaMalloc(&dQ, n_qkv * 4)); CUDA_CHECK(cudaMalloc(&dK, n_qkv * 4));
    CUDA_CHECK(cudaMalloc(&dV, n_qkv * 4)); CUDA_CHECK(cudaMalloc(&dO, n_qkv * 4));
    CUDA_CHECK(cudaMalloc(&dL, n_l * 4));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), n_qkv * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), n_qkv * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), n_qkv * 4, cudaMemcpyHostToDevice));

    float scale = 1.f / sqrtf((float)D);
    dim3 grid((N + BR - 1) / BR, H, B);
    flash_fwd<D, BR, BC, causal><<<grid, BR>>>(dQ, dK, dV, dO, dL, N, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, n_qkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL.data(), dL, n_l * 4, cudaMemcpyDeviceToHost));

    attention_cpu(hQ.data(), hK.data(), hV.data(), refO.data(), refL.data(),
                  B, H, N, D, scale, causal);

    float max_err = 0.f;
    for (size_t i = 0; i < n_qkv; i++) max_err = fmaxf(max_err, fabsf(hO[i] - refO[i]));
    float max_err_l = 0.f;
    for (size_t i = 0; i < n_l; i++) max_err_l = fmaxf(max_err_l, fabsf(hL[i] - refL[i]));

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);

    bool ok = max_err < 1e-4f && max_err_l < 1e-4f;
    printf("%-12s B=%d H=%d N=%-5d  max|dO|=%.2e  max|dL|=%.2e  %s\n",
           causal ? "causal" : "non-causal", B, H, N, max_err, max_err_l,
           ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_test<false>(2, 3, 200);   // N not a multiple of Br/Bc: edge guards
    ok &= run_test<false>(1, 2, 512);
    ok &= run_test<true >(2, 3, 200);
    ok &= run_test<true >(1, 2, 512);
    return ok ? 0 : 1;
}
