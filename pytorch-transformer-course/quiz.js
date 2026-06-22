/* PyTorch transformer course — navigation, problem-set engine, progress tracking */

const CHAPTERS = [
  { file: "index.html", title: "Course Home", short: "Home" },
  { file: "ch1.html", title: "1 · Tensors: Shapes, Broadcasting & In-Place Ops", short: "Ch 1" },
  { file: "ch2.html", title: "2 · Autograd: backward(), .grad & no_grad", short: "Ch 2" },
  { file: "ch3.html", title: "3 · nn.Module: Parameters, Buffers & Submodules", short: "Ch 3" },
  { file: "ch4.html", title: "4 · Data, Losses & the Training Loop", short: "Ch 4" },
  { file: "ch5.html", title: "5 · The Transformer as Tensor Ops", short: "Ch 5" },
  { file: "ch6.html", title: "6 · The Transformer in PyTorch", short: "Ch 6" },
  { file: "ch7.html", title: "7 · Training, Sampling & Debugging", short: "Ch 7" },
];

function pageName() {
  const p = location.pathname.split("/").pop();
  return p === "" ? "index.html" : p;
}

function quizKey(page, i) { return "pttf-progress:" + page + ":" + i; }
function totalKey(page) { return "pttf-total:" + page; }
function workKey(page, i) { return "pttf-work:" + page + ":" + i; }

/* ---------- sidebar ---------- */
function buildSidebar() {
  const cur = pageName();
  const sb = document.createElement("nav");
  sb.id = "sidebar";
  let html =
    '<a class="brand" href="index.html"><span class="bolt">&#128293;</span> A Transformer in PyTorch</a>' +
    '<div class="tagline">PyTorch syntax from zero to a trained language model</div>';
  for (const ch of CHAPTERS) {
    const total = parseInt(localStorage.getItem(totalKey(ch.file)) || "0", 10);
    let done = 0;
    for (let i = 0; i < total; i++) {
      if (localStorage.getItem(quizKey(ch.file, i)) === "1") done++;
    }
    const prog = total > 0 ? `<span class="prog">${done}/${total}</span>` : "";
    html += `<a class="chap ${ch.file === cur ? "current" : ""}" href="${ch.file}">${ch.title}${prog}</a>`;
  }
  sb.innerHTML = html;
  document.body.prepend(sb);
}

/* ---------- prev/next ---------- */
function buildPageNav() {
  const cur = pageName();
  const idx = CHAPTERS.findIndex((c) => c.file === cur);
  if (idx < 0) return;
  const nav = document.createElement("div");
  nav.className = "pagenav";
  let html = "";
  if (idx > 0) {
    const p = CHAPTERS[idx - 1];
    html += `<a class="prev" href="${p.file}"><div class="dir">&larr; Previous</div>${p.title}</a>`;
  }
  if (idx < CHAPTERS.length - 1) {
    const n = CHAPTERS[idx + 1];
    html += `<a class="next" href="${n.file}"><div class="dir">Next &rarr;</div>${n.title}</a>`;
  }
  nav.innerHTML = html;
  document.querySelector("main").appendChild(nav);
}

/* ---------- problem-set engine ----------
   Each .quiz block is one problem: a statement (.q-text), an editable code
   workspace (injected here, persisted to localStorage), progressive hints
   (.hint), and a reference solution (.answer) that stays locked until a
   genuine attempt exists in the workspace.
   Per-problem knobs: data-min (attempt length to unlock the reference,
   default 40 chars), data-placeholder (workspace placeholder text).       */
function enhanceQuizzes() {
  const page = pageName();
  const quizzes = document.querySelectorAll(".quiz");
  localStorage.setItem(totalKey(page), String(quizzes.length));

  quizzes.forEach((q, i) => {
    const label = document.createElement("div");
    label.className = "q-label";
    label.textContent = "Problem " + (i + 1);
    q.prepend(label);

    // workspace: where the solution gets written
    const ta = document.createElement("textarea");
    ta.spellcheck = false;
    ta.placeholder = q.dataset.placeholder ||
      "# Write your solution here — real code, then run it in a REPL.";
    ta.value = localStorage.getItem(workKey(page, i)) || "";
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const s = ta.selectionStart, t = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + "    " + ta.value.slice(t);
        ta.selectionStart = ta.selectionEnd = s + 4;
      }
    });
    const ws = document.createElement("div");
    ws.className = "workspace";
    ws.appendChild(ta);
    const qtext = q.querySelector(".q-text");
    (qtext || label).insertAdjacentElement("afterend", ws);

    const controls = document.createElement("div");
    controls.className = "q-controls";

    // progressive hints
    const hints = q.querySelectorAll(".hint");
    if (hints.length > 0) {
      let shown = 0;
      const hb = document.createElement("button");
      hb.className = "hint-btn";
      hb.textContent = `\u{1F4A1} Hint (0/${hints.length})`;
      hb.addEventListener("click", () => {
        if (shown < hints.length) {
          hints[shown].classList.add("shown");
          shown++;
          hb.textContent = `\u{1F4A1} Hint (${shown}/${hints.length})`;
          if (shown === hints.length) hb.disabled = true;
        }
      });
      controls.appendChild(hb);
    }

    // reference solution, gated on a real attempt in the workspace
    const ans = q.querySelector(".answer");
    const minLen = parseInt(q.dataset.min || "40", 10);
    let ab = null, gate = null;
    if (ans) {
      const lbl = document.createElement("div");
      lbl.className = "ans-label";
      lbl.textContent = "Reference solution";
      ans.prepend(lbl);
      ab = document.createElement("button");
      ab.className = "answer-btn";
      ab.textContent = "Compare with reference";
      ab.addEventListener("click", () => {
        ans.classList.add("shown");
        ab.style.display = "none";
        if (gate) gate.style.display = "none";
      });
      controls.appendChild(ab);
      gate = document.createElement("div");
      gate.className = "gate-note";
      gate.textContent =
        `\u{1F512} Reference unlocks after a real attempt (${minLen}+ characters in the box).`;
    }

    function updateGate() {
      if (!ab) return;
      const ok = ta.value.trim().length >= minLen;
      ab.disabled = !ok;
      if (gate) gate.style.display = ok ? "none" : "";
    }
    ta.addEventListener("input", () => {
      localStorage.setItem(workKey(page, i), ta.value);
      updateGate();
    });
    updateGate();

    // link back to the concept section
    const target = q.dataset.concept;
    if (target) {
      const a = document.createElement("a");
      a.className = "concept-link";
      a.href = target;
      a.textContent = "\u{1F4D6} Review: " + (q.dataset.conceptLabel || "concept");
      controls.appendChild(a);
    }

    // solved checkbox, persisted
    const wrap = document.createElement("label");
    wrap.className = "gotit";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = localStorage.getItem(quizKey(page, i)) === "1";
    cb.addEventListener("change", () => {
      localStorage.setItem(quizKey(page, i), cb.checked ? "1" : "0");
      updateProgressBar();
    });
    wrap.appendChild(cb);
    wrap.appendChild(document.createTextNode(" solved it (before opening the reference)"));
    controls.appendChild(wrap);

    q.appendChild(controls);
    if (gate) q.appendChild(gate);
  });

  // per-page progress bar
  if (quizzes.length > 0) {
    const bar = document.createElement("div");
    bar.id = "page-progress";
    bar.innerHTML =
      '<span class="prog-text"></span><div class="bar-outer"><div class="bar-inner"></div></div>';
    const h1 = document.querySelector("main h1");
    const sub = document.querySelector("main .subtitle");
    (sub || h1).insertAdjacentElement("afterend", bar);
    updateProgressBar();
  }
}

function updateProgressBar() {
  const page = pageName();
  const bar = document.getElementById("page-progress");
  if (!bar) return;
  const total = document.querySelectorAll(".quiz").length;
  let done = 0;
  for (let i = 0; i < total; i++) {
    if (localStorage.getItem(quizKey(page, i)) === "1") done++;
  }
  bar.querySelector(".prog-text").textContent =
    `Problem set: ${done} / ${total} solved on this page`;
  bar.querySelector(".bar-inner").style.width =
    total ? (100 * done) / total + "%" : "0";
}

/* ---------- index dashboard ---------- */
function buildDashboard() {
  const el = document.getElementById("dashboard");
  if (!el) return;
  let html = "";
  let grandDone = 0, grandTotal = 0;
  for (const ch of CHAPTERS) {
    if (ch.file === "index.html") continue;
    const total = parseInt(localStorage.getItem(totalKey(ch.file)) || "0", 10);
    let done = 0;
    for (let i = 0; i < total; i++) {
      if (localStorage.getItem(quizKey(ch.file, i)) === "1") done++;
    }
    grandDone += done; grandTotal += total;
    const pct = total ? (100 * done) / total : 0;
    const count = total ? `${done}/${total}` : "not visited";
    html += `<div class="dash-row"><span class="dash-title"><a href="${ch.file}">${ch.title}</a></span>` +
      `<span class="dash-bar"><span class="dash-fill" style="width:${pct}%; display:block"></span></span>` +
      `<span class="dash-count">${count}</span></div>`;
  }
  if (grandTotal > 0) {
    html += `<p style="margin-top:0.8rem; color: var(--ink-soft); font-size:0.9rem">` +
      `Total: <strong>${grandDone}/${grandTotal}</strong> problems solved. ` +
      `Come back to unsolved ones tomorrow, then in three days — spacing beats cramming.</p>`;
  }
  el.innerHTML = html;
}

document.addEventListener("DOMContentLoaded", () => {
  buildSidebar();
  enhanceQuizzes();
  buildPageNav();
  buildDashboard();
});
