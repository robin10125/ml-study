// flash_bwd.cu — FlashAttention-2-style backward pass, teaching version.
// Column-parallel (one thread block per K/V tile), recomputes P from L,
// accumulates dK/dV locally and dQ via atomicAdd. Verified vs CPU gradients.
//
// Build:  nvcc -O3 -arch=native -o flash_bwd flash_bwd.cu
// Run:    ./flash_bwd
//
// Layout: Q,K,V,O,dO,dQ,dK,dV are [B, H, N, d]; L and D are [B, H, N].

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
// Forward kernel (from Chapter 5) — needed to produce O and L for backward.
// ---------------------------------------------------------------------------
template <int d, int Br, int Bc, bool causal>
__global__ void flash_fwd(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O,
                          float* __restrict__ L,
                          int N, float scale) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x * Br + tid;
    const size_t bh = (size_t)blockIdx.z * gridDim.y + blockIdx.y;
    const float* Qh = Q + bh * N * d;
    const float* Kh = K + bh * N * d;
    const float* Vh = V + bh * N * d;
    float*       Oh = O + bh * N * d;
    float*       Lh = L + bh * N;

    __shared__ float Ks[Bc][d];
    __shared__ float Vs[Bc][d];
    float q[d], o[d], s[Bc];
    float m = -INFINITY, l = 0.f;

    if (row < N)
        for (int k = 0; k < d; k++) q[k] = Qh[(size_t)row * d + k];
    for (int k = 0; k < d; k++) o[k] = 0.f;

    const int j_end = causal ? min(N, blockIdx.x * Br + Br) : N;
    for (int j0 = 0; j0 < j_end; j0 += Bc) {
        for (int idx = tid; idx < Bc * d; idx += Br) {
            int r = idx / d, c = idx % d;
            bool in_range = (j0 + r) < N;
            Ks[r][c] = in_range ? Kh[(size_t)(j0 + r) * d + c] : 0.f;
            Vs[r][c] = in_range ? Vh[(size_t)(j0 + r) * d + c] : 0.f;
        }
        __syncthreads();
        if (row < N) {
            float mt = -INFINITY;
            for (int c = 0; c < Bc; c++) {
                float dot = 0.f;
#pragma unroll
                for (int k = 0; k < d; k++) dot += q[k] * Ks[c][k];
                bool masked = (j0 + c >= N) || (causal && j0 + c > row);
                s[c] = masked ? -INFINITY : dot * scale;
                mt = fmaxf(mt, s[c]);
            }
            float m_new = fmaxf(m, mt);
            float alpha = __expf(m - m_new);
            l *= alpha;
#pragma unroll
            for (int k = 0; k < d; k++) o[k] *= alpha;
            for (int c = 0; c < Bc; c++) {
                float p = __expf(s[c] - m_new);
                l += p;
#pragma unroll
                for (int k = 0; k < d; k++) o[k] += p * Vs[c][k];
            }
            m = m_new;
        }
        __syncthreads();
    }
    if (row < N) {
        float inv = 1.f / l;
        for (int k = 0; k < d; k++) Oh[(size_t)row * d + k] = o[k] * inv;
        Lh[row] = m + logf(l);
    }
}

// ---------------------------------------------------------------------------
// Backward preprocess: D_i = sum_k dO_ik * O_ik.  Grid: (ceil(N/256), H, B).
// ---------------------------------------------------------------------------
template <int d>
__global__ void bwd_preprocess(const float* __restrict__ dO,
                               const float* __restrict__ O,
                               float* __restrict__ D, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    size_t bh = (size_t)blockIdx.z * gridDim.y + blockIdx.y;
    const float* dOh = dO + bh * N * d + (size_t)row * d;
    const float* Oh  = O  + bh * N * d + (size_t)row * d;
    float acc = 0.f;
    for (int k = 0; k < d; k++) acc += dOh[k] * Oh[k];
    D[bh * N + row] = acc;
}

// ---------------------------------------------------------------------------
// Backward kernel. Grid: (ceil(N/Bc), H, B). Block: Bc threads, one per
// key/value row ("column" of S). dK, dV accumulate locally; dQ via atomicAdd.
// dQ must be zeroed before launch.
// ---------------------------------------------------------------------------
template <int d, int Br, int Bc, bool causal>
__global__ void flash_bwd(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          const float* __restrict__ dO,
                          const float* __restrict__ L,
                          const float* __restrict__ D,
                          float* __restrict__ dQ,
                          float* __restrict__ dK,
                          float* __restrict__ dV,
                          int N, float scale) {
    const int tid = threadIdx.x;
    const int col = blockIdx.x * Bc + tid;          // my key/value row
    const size_t bh = (size_t)blockIdx.z * gridDim.y + blockIdx.y;
    const float* Qh  = Q  + bh * N * d;
    const float* Kh  = K  + bh * N * d;
    const float* Vh  = V  + bh * N * d;
    const float* dOh = dO + bh * N * d;
    const float* Lh  = L  + bh * N;
    const float* Dh  = D  + bh * N;
    float* dQh = dQ + bh * N * d;
    float* dKh = dK + bh * N * d;
    float* dVh = dV + bh * N * d;

    __shared__ float Qs[Br][d];
    __shared__ float dOs[Br][d];
    __shared__ float Ls[Br], Ds[Br];

    // My column's k, v and gradient accumulators (large: spills are expected
    // in this teaching version; real FA-2 tiles these through tensor cores).
    float kvec[d], vvec[d], dk[d], dv[d];
    if (col < N)
        for (int x = 0; x < d; x++) {
            kvec[x] = Kh[(size_t)col * d + x];
            vvec[x] = Vh[(size_t)col * d + x];
        }
    for (int x = 0; x < d; x++) { dk[x] = 0.f; dv[x] = 0.f; }

    // Causal: query blocks entirely before my columns see none of them.
    const int i_start = causal ? (blockIdx.x * Bc / Br) * Br : 0;

    for (int i0 = i_start; i0 < N; i0 += Br) {
        for (int idx = tid; idx < Br * d; idx += Bc) {
            int r = idx / d, c = idx % d;
            bool in_range = (i0 + r) < N;
            Qs[r][c]  = in_range ? Qh [(size_t)(i0 + r) * d + c] : 0.f;
            dOs[r][c] = in_range ? dOh[(size_t)(i0 + r) * d + c] : 0.f;
        }
        for (int idx = tid; idx < Br; idx += Bc) {
            bool in_range = (i0 + idx) < N;
            Ls[idx] = in_range ? Lh[i0 + idx] : 0.f;
            Ds[idx] = in_range ? Dh[i0 + idx] : 0.f;
        }
        __syncthreads();

        if (col < N) {
            int r_max = min(Br, N - i0);
            for (int r = 0; r < r_max; r++) {
                int qrow = i0 + r;
                if (causal && qrow < col) continue;     // future query? impossible; skip

                // Recompute one score and one attention weight from L (Ch. 3 §3.5).
                float s = 0.f;
#pragma unroll
                for (int x = 0; x < d; x++) s += Qs[r][x] * kvec[x];
                float p = __expf(s * scale - Ls[r]);    // P_rc, exact

                // dV_c += P_rc * dO_r ;  dP_rc = dO_r . v_c
                float dp = 0.f;
#pragma unroll
                for (int x = 0; x < d; x++) {
                    dv[x] += p * dOs[r][x];
                    dp    += dOs[r][x] * vvec[x];
                }

                // dS_rc = P_rc * (dP_rc - D_r); fold in the 1/sqrt(d) of S.
                float ds = p * (dp - Ds[r]) * scale;

                // dK_c += dS_rc * q_r ;  dQ_r += dS_rc * k_c (cross-block: atomic)
#pragma unroll
                for (int x = 0; x < d; x++) {
                    dk[x] += ds * Qs[r][x];
                    atomicAdd(&dQh[(size_t)qrow * d + x], ds * kvec[x]);
                }
            }
        }
        __syncthreads();
    }

    if (col < N)
        for (int x = 0; x < d; x++) {
            dKh[(size_t)col * d + x] = dk[x];
            dVh[(size_t)col * d + x] = dv[x];
        }
}

// ---------------------------------------------------------------------------
// CPU reference gradients (materializes P — fine for testing).
// ---------------------------------------------------------------------------
void attention_bwd_cpu(const float* Q, const float* K, const float* V,
                       const float* dO,
                       float* dQ, float* dK, float* dV,
                       int B, int H, int N, int d, float scale, bool causal) {
    std::vector<float> S((size_t)N * N), P((size_t)N * N), dP((size_t)N * N);
    std::vector<float> O((size_t)N * d), Dv(N);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            size_t bh = ((size_t)b * H + h);
            const float* Qh = Q + bh * N * d;   const float* Kh = K + bh * N * d;
            const float* Vh = V + bh * N * d;   const float* dOh = dO + bh * N * d;
            float* dQh = dQ + bh * N * d;       float* dKh = dK + bh * N * d;
            float* dVh = dV + bh * N * d;

            for (int i = 0; i < N; i++) {
                float m = -INFINITY;
                for (int j = 0; j < N; j++) {
                    float dot = 0.f;
                    for (int k = 0; k < d; k++) dot += Qh[i*d+k] * Kh[j*d+k];
                    S[(size_t)i*N+j] = (causal && j > i) ? -INFINITY : dot * scale;
                    m = fmaxf(m, S[(size_t)i*N+j]);
                }
                float l = 0.f;
                for (int j = 0; j < N; j++) { P[(size_t)i*N+j] = expf(S[(size_t)i*N+j] - m); l += P[(size_t)i*N+j]; }
                for (int j = 0; j < N; j++) P[(size_t)i*N+j] /= l;
            }
            for (int i = 0; i < N; i++)
                for (int k = 0; k < d; k++) {
                    float acc = 0.f;
                    for (int j = 0; j < N; j++) acc += P[(size_t)i*N+j] * Vh[j*d+k];
                    O[(size_t)i*d+k] = acc;
                }
            for (int i = 0; i < N; i++) {
                float acc = 0.f;
                for (int k = 0; k < d; k++) acc += dOh[i*d+k] * O[(size_t)i*d+k];
                Dv[i] = acc;
            }
            // dV = P^T dO
            for (int j = 0; j < N; j++)
                for (int k = 0; k < d; k++) {
                    float acc = 0.f;
                    for (int i = 0; i < N; i++) acc += P[(size_t)i*N+j] * dOh[i*d+k];
                    dVh[j*d+k] = acc;
                }
            // dP = dO V^T ; dS = P o (dP - D) * scale
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    float acc = 0.f;
                    for (int k = 0; k < d; k++) acc += dOh[i*d+k] * Vh[j*d+k];
                    dP[(size_t)i*N+j] = P[(size_t)i*N+j] * (acc - Dv[i]) * scale;
                }
            // dQ = dS K ; dK = dS^T Q
            for (int i = 0; i < N; i++)
                for (int k = 0; k < d; k++) {
                    float acc = 0.f;
                    for (int j = 0; j < N; j++) acc += dP[(size_t)i*N+j] * Kh[j*d+k];
                    dQh[i*d+k] = acc;
                }
            for (int j = 0; j < N; j++)
                for (int k = 0; k < d; k++) {
                    float acc = 0.f;
                    for (int i = 0; i < N; i++) acc += dP[(size_t)i*N+j] * Qh[i*d+k];
                    dKh[j*d+k] = acc;
                }
        }
}

// ---------------------------------------------------------------------------
// Test harness.
// ---------------------------------------------------------------------------
constexpr int D = 64, BR = 64, BC = 64;

template <bool causal>
bool run_test(int B, int H, int N) {
    size_t n = (size_t)B * H * N * D, n_l = (size_t)B * H * N;
    std::vector<float> hQ(n), hK(n), hV(n), hdO(n);
    std::vector<float> hdQ(n), hdK(n), hdV(n), rdQ(n), rdK(n), rdV(n);
    srand(123);
    for (auto* v : {&hQ, &hK, &hV, &hdO})
        for (auto& x : *v) x = 2.f * rand() / RAND_MAX - 1.f;

    float *dQ_, *dK_, *dV_, *dO_, *dL_, *ddO, *dD, *ddQ, *ddK, *ddV;
    for (auto p : {&dQ_, &dK_, &dV_, &dO_, &ddO, &ddQ, &ddK, &ddV})
        CUDA_CHECK(cudaMalloc(p, n * 4));
    CUDA_CHECK(cudaMalloc(&dL_, n_l * 4));
    CUDA_CHECK(cudaMalloc(&dD, n_l * 4));
    CUDA_CHECK(cudaMemcpy(dQ_, hQ.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK_, hK.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV_, hV.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ddO, hdO.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ddQ, 0, n * 4));   // atomicAdd target must start at 0

    float scale = 1.f / sqrtf((float)D);

    dim3 grid_fwd((N + BR - 1) / BR, H, B);
    flash_fwd<D, BR, BC, causal><<<grid_fwd, BR>>>(dQ_, dK_, dV_, dO_, dL_, N, scale);

    dim3 grid_pre((N + 255) / 256, H, B);
    bwd_preprocess<D><<<grid_pre, 256>>>(ddO, dO_, dD, N);

    dim3 grid_bwd((N + BC - 1) / BC, H, B);
    flash_bwd<D, BR, BC, causal><<<grid_bwd, BC>>>(dQ_, dK_, dV_, ddO, dL_, dD,
                                                   ddQ, ddK, ddV, N, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hdQ.data(), ddQ, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdK.data(), ddK, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdV.data(), ddV, n * 4, cudaMemcpyDeviceToHost));

    attention_bwd_cpu(hQ.data(), hK.data(), hV.data(), hdO.data(),
                      rdQ.data(), rdK.data(), rdV.data(),
                      B, H, N, D, scale, causal);

    auto max_err = [&](std::vector<float>& a, std::vector<float>& b) {
        float e = 0.f;
        for (size_t i = 0; i < a.size(); i++) e = fmaxf(e, fabsf(a[i] - b[i]));
        return e;
    };
    float eQ = max_err(hdQ, rdQ), eK = max_err(hdK, rdK), eV = max_err(hdV, rdV);

    for (auto p : {dQ_, dK_, dV_, dO_, dL_, ddO, dD, ddQ, ddK, ddV}) cudaFree(p);

    bool ok = eQ < 2e-4f && eK < 2e-4f && eV < 2e-4f;
    printf("%-12s B=%d H=%d N=%-5d  max|ddQ|=%.2e  max|ddK|=%.2e  max|ddV|=%.2e  %s\n",
           causal ? "causal" : "non-causal", B, H, N, eQ, eK, eV, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_test<false>(2, 2, 200);
    ok &= run_test<false>(1, 2, 384);
    ok &= run_test<true >(2, 2, 200);
    ok &= run_test<true >(1, 2, 384);
    return ok ? 0 : 1;
}
