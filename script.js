// ════════════════════════════════════════════════════════════════
// N-grams — Unigram, Bigram, Trigram
// Single continuous page, vanilla JS, same design language as the
// Sentiment Analysis and HMM/CRF pages.
// ════════════════════════════════════════════════════════════════

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ─── DATA ────────────────────────────────────────────────────────────────
const N_META = {
  1: { label:"Unigram", color:"var(--uni)", bg:"var(--uni-bg)", bd:"var(--uni-bd)", cls:"uni", tabCls:"sel-1", badge:"b-uni", chip:"c-uni", insight:"i-uni", desc:"Looks at one word at a time — no context at all." },
  2: { label:"Bigram",  color:"var(--bi)",  bg:"var(--bi-bg)",  bd:"var(--bi-bd)",  cls:"bi",  tabCls:"sel-2", badge:"b-bi",  chip:"c-bi",  insight:"i-bi",  desc:"Looks at pairs of consecutive words — one word of context." },
  3: { label:"Trigram", color:"var(--tri)", bg:"var(--tri-bg)", bd:"var(--tri-bd)", cls:"tri", tabCls:"sel-3", badge:"b-tri", chip:"c-tri", insight:"i-tri", desc:"Looks at three consecutive words — two words of context." },
};

const BASICS_SENTENCE = "the quick brown fox jumps over the lazy dog";
const PLAYGROUND_DEFAULT = "she sells seashells by the seashore";

const CORPUS_TEXT = "language models predict the next word in a sentence . a good language model learns patterns from large amounts of text . n gram models count how often word sequences occur . bigram models look at pairs of words while trigram models look at three words at a time . the more context a model uses the better its predictions can be . however more context also means more data is needed to avoid sparsity . modern language models use neural networks instead of simple counts . but the core idea of predicting the next word remains the same .";

const NGRAM_LIMITATIONS = [
  "Data sparsity: most possible word sequences are never seen in training, so their probability is zero.",
  "Fixed, short window: only the previous N−1 words count — anything further back is invisible to the model.",
  "No real understanding: it's pure counting, not meaning — it can't tell \"bank\" (river) from \"bank\" (money).",
  "Storage grows fast: more N or more vocabulary means exponentially more possible sequences to store.",
];

const NGRAM_USES = [
  "Autocomplete and predictive text keyboards.",
  "Spelling and grammar correction.",
  "Speech recognition language modeling.",
  "Machine translation quality scoring (e.g. the BLEU metric compares n-gram overlap).",
  "Plagiarism and duplicate-content detection.",
  "Search engines ranking how well a query matches a document.",
];

const QUIZ = [
  { id:1, tag:"Basics", q:"What does an n-gram model assume about predicting the next word?", opts:["It depends on the entire document read so far","It depends only on the previous N−1 words","It depends on random chance","It depends only on the last full sentence"], ans:1, exp:"N-gram models make a locality assumption: the next word's probability depends only on a fixed, small window of previous words (N−1 of them), not the whole document. This is the same kind of simplifying assumption behind Markov models." },
  { id:2, tag:"Tradeoff", q:"Why doesn't a language model just use a huge N, like N=10, for maximum context?", opts:["Higher N always makes predictions worse","Long exact word sequences are extremely rare in any real corpus, so most counts end up at zero (data sparsity)","Computers cannot store more than 3 words at once","N above 3 is not mathematically valid"], ans:1, exp:"As N grows, the number of possible word sequences explodes, but any real corpus is finite. Most long sequences are never seen, so their counts are zero — this is the classic sparsity problem, and it's why small N (2 or 3) is often more practical than large N." },
  { id:3, tag:"Basics", q:"How many bigrams are there in the 5-word sentence \"the cat sat on it\"?", opts:["5","4","3","25"], ans:1, exp:"Number of n-grams = (sentence length) − N + 1. For 5 words and N=2: 5 − 2 + 1 = 4 bigrams: \"the cat\", \"cat sat\", \"sat on\", \"on it\"." },
  { id:4, tag:"Uses", q:"Which of these is NOT a typical use of n-gram models?", opts:["Autocomplete / predictive text", "Spelling and grammar correction", "Detecting objects in a photograph", "Speech recognition language modeling"], ans:2, exp:"N-grams operate over sequences of discrete tokens like words or characters. Detecting objects in an image is a computer vision task working over pixels, not a typical n-gram application (even though a loosely similar \"sliding window\" idea shows up there too)." },
  { id:5, tag:"Basics", q:"In a bigram model, the word \"cat\" appears 10 times in training, and is followed by \"sat\" 3 of those times. What is P(sat | cat)?", opts:["3", "0.3", "10", "30"], ans:1, exp:"P(next word | context) = count(context, next word) / count(context) = 3 / 10 = 0.3. This is exactly the calculation behind the probability bars in the generator above — count how often a word follows the context, divide by the context's total count." },
  { id:6, tag:"Tradeoff", q:"For a vocabulary of size V, roughly how many possible distinct n-grams are there as N increases?", opts:["It grows exponentially (on the order of V^N)", "It grows linearly with N", "It stays the same no matter what N is", "It shrinks as N increases"], ans:0, exp:"Every extra word in the window multiplies the number of possible combinations by roughly V again, so the space of possible n-grams grows exponentially with N. That's the mathematical root of both the sparsity problem and the storage-growth limitation." },
];

// ─── HELPERS ─────────────────────────────────────────────────────────────
function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s'-]/g, " ").split(/\s+/).filter(Boolean);
}

function getNgrams(tokens, n) {
  const grams = [];
  for (let i = 0; i <= tokens.length - n; i++) {
    const words = tokens.slice(i, i + n);
    grams.push({ start: i, words, phrase: words.join(" ") });
  }
  return grams;
}

function buildModel(tokens, n) {
  const model = {};
  for (let i = 0; i <= tokens.length - n; i++) {
    const context = n === 1 ? "" : tokens.slice(i, i + n - 1).join(" ");
    const nextWord = tokens[i + n - 1];
    if (!model[context]) model[context] = {};
    model[context][nextWord] = (model[context][nextWord] || 0) + 1;
  }
  return model;
}

function candidatesFor(model, context) {
  const counts = model[context];
  if (!counts) return null;
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return Object.entries(counts)
    .map(([word, count]) => ({ word, count, pct: Math.round((count / total) * 100) }))
    .sort((a, b) => b.count - a.count);
}

// ─── STATE ───────────────────────────────────────────────────────────────
const CORPUS_TOKENS = tokenize(CORPUS_TEXT);
const GEN_MODELS = { 1: buildModel(CORPUS_TOKENS, 1), 2: buildModel(CORPUS_TOKENS, 2), 3: buildModel(CORPUS_TOKENS, 3) };
const BIGRAM_SEEDS = ["the", "language", "models", "context", "word", "more"];
const TRIGRAM_SEEDS = ["language models", "the next", "n gram", "bigram models", "more context", "of the"];

const STATE = {
  basics: { n: 2, pos: 0, sentence: BASICS_SENTENCE },
  playground: { n: 2, sentence: PLAYGROUND_DEFAULT },
  gen: { n: 2, sequence: [] },
};

// ─── APP INIT ────────────────────────────────────────────────────────────
function initApp() {
  document.getElementById("app").innerHTML = `
    <div class="sv-shell">
      <div class="sv-hero">
        <div class="sv-hero-pillrow">
          <span class="sv-hero-pill">Foundations</span>
          <span class="sv-hero-pillsub">Language Modeling</span>
        </div>
        <h1>N-grams: Unigram, Bigram, Trigram</h1>
        <p>The simplest way to teach a computer to guess the next word — by counting which words tend to follow which. Click through the sections below to see it in action.</p>
      </div>
      <div class="sv-maininner" id="mainInner"></div>
    </div>
  `;
  document.getElementById("mainInner").innerHTML = `
    <div id="secBasics"></div>
    <div id="secPlayground"></div>
    <div id="secGenerator"></div>
    <div id="secLimits"></div>
    <div id="secQuiz"></div>
  `;
  mountBasics();
  mountPlayground();
  mountGenerator();
  mountLimitsUses();
  mountQuiz();
}

document.addEventListener("DOMContentLoaded", initApp);

// ─── SECTION: BASICS (sliding window visualizer) ───────────────────────────
function mountBasics() {
  document.getElementById("secBasics").innerHTML = `
    <div class="sv-card">
      <div class="sv-badge b-bas">Start here</div>
      <div class="sv-card-title">What's An N-gram?</div>
      <p class="sv-desc">An n-gram is just a run of N consecutive words. "N" is a number you choose — 1 word (unigram), 2 words (bigram), 3 words (trigram), and so on. Pick a size below, then step through the sentence and watch the window slide.</p>
      <div class="n-tabs" id="basicsNTabs"></div>
      <div class="token-row" id="basicsTokenRow"></div>
      <div class="window-nav">
        <button class="sv-btn btn-sec" id="basicsPrevBtn" onclick="basicsStep(-1)">← Prev</button>
        <span class="pos-label" id="basicsPosLabel"></span>
        <button class="sv-btn btn-sec" id="basicsNextBtn" onclick="basicsStep(1)">Next →</button>
      </div>
      <div class="current-gram">
        <div class="current-gram-label" id="basicsCurLabel"></div>
        <div class="current-gram-value" id="basicsCurValue"></div>
      </div>
      <div class="sv-desc" style="margin-bottom:6px;">All the n-grams in this sentence:</div>
      <div class="word-grid" id="basicsAllGrams"></div>
      <div class="sv-insight" id="basicsInsight"></div>
    </div>
  `;
  document.getElementById("basicsNTabs").innerHTML = [1, 2, 3].map(n => {
    const m = N_META[n];
    const sel = STATE.basics.n === n ? ` ${m.tabCls}` : "";
    return `<button class="n-tab${sel}" onclick="setBasicsN(${n})">${m.label}</button>`;
  }).join("");
  renderBasicsWindow();
}

function setBasicsN(n) {
  STATE.basics.n = n;
  STATE.basics.pos = 0;
  document.getElementById("basicsNTabs").innerHTML = [1, 2, 3].map(k => {
    const m = N_META[k];
    const sel = STATE.basics.n === k ? ` ${m.tabCls}` : "";
    return `<button class="n-tab${sel}" onclick="setBasicsN(${k})">${m.label}</button>`;
  }).join("");
  renderBasicsWindow();
}
window.setBasicsN = setBasicsN;

function basicsStep(delta) {
  const tokens = tokenize(STATE.basics.sentence);
  const maxPos = tokens.length - STATE.basics.n;
  STATE.basics.pos = Math.max(0, Math.min(maxPos, STATE.basics.pos + delta));
  renderBasicsWindow();
}
window.basicsStep = basicsStep;

function jumpBasicsPos(i) {
  STATE.basics.pos = i;
  renderBasicsWindow();
}
window.jumpBasicsPos = jumpBasicsPos;

function renderBasicsWindow() {
  const n = STATE.basics.n;
  const m = N_META[n];
  const tokens = tokenize(STATE.basics.sentence);
  const maxPos = tokens.length - n;
  const pos = STATE.basics.pos;
  const grams = getNgrams(tokens, n);

  document.getElementById("basicsTokenRow").innerHTML = tokens.map((word, i) => {
    const inWindow = i >= pos && i < pos + n;
    return `<div class="token-box${inWindow ? " in-window" : ""}">${escapeHtml(word)}</div>`;
  }).join("");

  document.getElementById("basicsPosLabel").textContent = `Window ${pos + 1} of ${maxPos + 1}`;
  document.getElementById("basicsPrevBtn").disabled = pos === 0;
  document.getElementById("basicsNextBtn").disabled = pos === maxPos;

  document.getElementById("basicsCurLabel").innerHTML = `<span class="chip ${m.chip}">${m.label}</span> at this position`;
  document.getElementById("basicsCurValue").textContent = `"${grams[pos].phrase}"`;

  document.getElementById("basicsAllGrams").innerHTML = grams.map((g, i) => {
    const active = i === pos;
    return `<button class="chip ${m.chip} c-mono" style="cursor:pointer;${active ? "outline:1.5px solid " + m.color + ";outline-offset:1px;font-weight:700;" : ""}" onclick="jumpBasicsPos(${i})">${escapeHtml(g.phrase)}</button>`;
  }).join("");

  document.getElementById("basicsInsight").className = `sv-insight ${m.insight}`;
  document.getElementById("basicsInsight").innerHTML = `💡 ${m.desc} A sentence of ${tokens.length} words has <strong>${grams.length}</strong> ${m.label.toLowerCase()}s (that's length − N + 1).`;
}

// ─── SECTION: PLAYGROUND (free-text extraction) ────────────────────────────
function mountPlayground() {
  document.getElementById("secPlayground").innerHTML = `
    <div class="sv-eyebrow">Try with Your Own Sentence</div>
    <div class="sv-card">
      <div class="sv-card-title">N-gram Playground</div>
      <p class="sv-desc">Type any sentence, pick a size, and see every n-gram it contains — instantly.</p>
      <div class="n-tabs" id="pgNTabs"></div>
      <input class="sv-input" id="pgSentence" style="margin-bottom:14px;">
      <div class="metric-row" id="pgMetrics"></div>
      <div class="word-grid" id="pgGrams"></div>
      <div class="sv-insight i-bas" id="pgInsight" style="display:none;"></div>
    </div>
  `;
  document.getElementById("pgNTabs").innerHTML = [1, 2, 3].map(n => {
    const m = N_META[n];
    const sel = STATE.playground.n === n ? ` ${m.tabCls}` : "";
    return `<button class="n-tab${sel}" onclick="setPgN(${n})">${m.label}</button>`;
  }).join("");
  const input = document.getElementById("pgSentence");
  input.value = STATE.playground.sentence;
  input.addEventListener("input", e => { STATE.playground.sentence = e.target.value; renderPlayground(); });
  renderPlayground();
}

function setPgN(n) {
  STATE.playground.n = n;
  document.getElementById("pgNTabs").innerHTML = [1, 2, 3].map(k => {
    const m = N_META[k];
    const sel = STATE.playground.n === k ? ` ${m.tabCls}` : "";
    return `<button class="n-tab${sel}" onclick="setPgN(${k})">${m.label}</button>`;
  }).join("");
  renderPlayground();
}
window.setPgN = setPgN;

function renderPlayground() {
  const n = STATE.playground.n;
  const m = N_META[n];
  const tokens = tokenize(STATE.playground.sentence);
  const insightEl = document.getElementById("pgInsight");

  if (tokens.length < n) {
    document.getElementById("pgMetrics").innerHTML = "";
    document.getElementById("pgGrams").innerHTML = "";
    insightEl.style.display = "block";
    insightEl.className = "sv-insight i-warn";
    insightEl.textContent = `Type at least ${n} word${n > 1 ? "s" : ""} to see ${m.label.toLowerCase()}s.`;
    return;
  }
  insightEl.style.display = "none";

  const grams = getNgrams(tokens, n);
  const uniqueGrams = new Set(grams.map(g => g.phrase)).size;

  document.getElementById("pgMetrics").innerHTML = `
    <div class="metric"><div class="metric-lbl">Words</div><div class="metric-val">${tokens.length}</div></div>
    <div class="metric"><div class="metric-lbl">${m.label}s</div><div class="metric-val" style="color:${m.color};">${grams.length}</div></div>
    <div class="metric"><div class="metric-lbl">Unique</div><div class="metric-val">${uniqueGrams}</div></div>
  `;
  document.getElementById("pgGrams").innerHTML = grams.map(g => `<span class="chip ${m.chip} c-mono">${escapeHtml(g.phrase)}</span>`).join("");
}

// ─── SECTION: GENERATOR (next-word predictor) ──────────────────────────────
function mountGenerator() {
  document.getElementById("secGenerator").innerHTML = `
    <div class="sv-eyebrow">From Counting To Predicting</div>
    <div class="sv-card">
      <div class="sv-card-title">Build A Tiny Language Model</div>
      <p class="sv-desc">We counted n-grams in a short paragraph about language models itself. Now use those counts to predict what word comes next — click a candidate to keep generating.</p>
      <div class="n-tabs" id="genNTabs"></div>
      <div id="genSeedRow"></div>
      <div class="gen-output" id="genOutput"></div>
      <div id="genCandidates"></div>
      <div style="text-align:right;margin-top:8px;">
        <button class="sv-btn btn-sec" onclick="resetGen()">↺ Reset</button>
      </div>
      <div class="sv-insight i-bas" style="margin-top:12px;">💡 This whole model is built from just <strong>${CORPUS_TOKENS.length} words</strong> of training text — tiny by design, so you can see exactly why some predictions are confident and others are missing entirely (that's the sparsity problem from the quiz below, live).</div>
    </div>
  `;
  document.getElementById("genNTabs").innerHTML = [1, 2, 3].map(n => {
    const m = N_META[n];
    const sel = STATE.gen.n === n ? ` ${m.tabCls}` : "";
    return `<button class="n-tab${sel}" onclick="setGenN(${n})">${m.label}</button>`;
  }).join("");
  resetGen();
}

function setGenN(n) {
  STATE.gen.n = n;
  STATE.gen.sequence = [];
  document.getElementById("genNTabs").innerHTML = [1, 2, 3].map(k => {
    const m = N_META[k];
    const sel = STATE.gen.n === k ? ` ${m.tabCls}` : "";
    return `<button class="n-tab${sel}" onclick="setGenN(${k})">${m.label}</button>`;
  }).join("");
  renderGenSeedRow();
  renderGenOutput();
  renderGenCandidates();
}
window.setGenN = setGenN;

function resetGen() {
  STATE.gen.sequence = [];
  renderGenSeedRow();
  renderGenOutput();
  renderGenCandidates();
}
window.resetGen = resetGen;

function pickSeed(phrase) {
  STATE.gen.sequence = phrase.split(" ");
  renderGenOutput();
  renderGenCandidates();
}
window.pickSeed = pickSeed;

function renderGenSeedRow() {
  const n = STATE.gen.n;
  const el = document.getElementById("genSeedRow");
  if (n === 1) { el.innerHTML = ""; return; }
  const seeds = n === 2 ? BIGRAM_SEEDS : TRIGRAM_SEEDS;
  el.innerHTML = `
    <div class="sv-desc" style="margin-bottom:6px;">Pick a starting ${n === 2 ? "word" : "phrase"}:</div>
    <div class="seed-row">${seeds.map(s => `<button class="chip c-gray c-mono" style="cursor:pointer;" onclick="pickSeed('${s.replace(/'/g, "\\'")}')">${escapeHtml(s)}</button>`).join("")}</div>
  `;
}

function currentGenContext() {
  const n = STATE.gen.n;
  if (n === 1) return "";
  const seq = STATE.gen.sequence;
  if (seq.length < n - 1) return null;
  return seq.slice(-(n - 1)).join(" ");
}

function renderGenOutput() {
  const seq = STATE.gen.sequence;
  const out = document.getElementById("genOutput");
  if (seq.length === 0) {
    out.innerHTML = `<span style="font-size:12px;color:#8a8878;">${STATE.gen.n === 1 ? "Click a word below to start generating." : "Pick a starting word above to begin."}</span>`;
    return;
  }
  out.innerHTML = seq.map((w, i) => `<span class="gen-word${i < STATE.gen.n - 1 && STATE.gen.n > 1 ? " seed" : ""}">${escapeHtml(w)}</span>`).join("") + `<span class="gen-cursor">▌</span>`;
}

function appendGenWord(word) {
  STATE.gen.sequence.push(word);
  renderGenOutput();
  renderGenCandidates();
}
window.appendGenWord = appendGenWord;

function renderGenCandidates() {
  const n = STATE.gen.n;
  const m = N_META[n];
  const container = document.getElementById("genCandidates");
  const context = currentGenContext();

  if (context === null) {
    container.innerHTML = "";
    return;
  }

  const candidates = candidatesFor(GEN_MODELS[n], context);

  if (!candidates) {
    container.innerHTML = `<div class="sv-insight i-warn">⚠️ No training data for the context "<strong>${escapeHtml(context)}</strong>" — this exact ${n === 2 ? "word" : "phrase"} was never followed by anything in our tiny corpus. This is exactly why real language models need smoothing or a much bigger corpus.</div>`;
    return;
  }

  const label = n === 1 ? "Most frequent words overall (no context used):" : `Words that followed "<strong>${escapeHtml(context)}</strong>" in training:`;
  container.innerHTML = `
    <div class="sv-desc" style="margin-bottom:8px;">${label}</div>
    ${candidates.slice(0, 6).map(c => `
      <button class="candidate-row" onclick="appendGenWord('${c.word.replace(/'/g, "\\'")}')">
        <span class="candidate-word">${escapeHtml(c.word)}</span>
        <div class="candidate-track"><div class="candidate-fill" style="width:${c.pct}%;background:${m.color};"></div></div>
        <span class="candidate-pct">${c.pct}%</span>
      </button>`).join("")}
  `;
}

// ─── SECTION: LIMITATIONS & USES ────────────────────────────────────────
function mountLimitsUses() {
  document.getElementById("secLimits").innerHTML = `
    <div class="sv-eyebrow">The Trade-offs</div>
    <div class="sv-card">
      <div class="sv-card-title">Limitations &amp; Uses</div>
      <p class="sv-desc">Counting words is fast and simple — but it comes with real limits, which is exactly why n-grams are still worth understanding even now that neural models exist.</p>
      <div class="limits-grid">
        <div>
          <div style="font-size:11px;font-weight:600;color:var(--neg);margin-bottom:8px;text-transform:uppercase;letter-spacing:.3px;">Limitations</div>
          ${NGRAM_LIMITATIONS.map(t => `<div style="display:flex;gap:6px;margin-bottom:8px;align-items:flex-start;"><div style="width:6px;height:6px;border-radius:50%;background:var(--neg);flex-shrink:0;margin-top:5px;"></div><div style="font-size:12px;color:#5a5950;line-height:1.55;">${escapeHtml(t)}</div></div>`).join("")}
        </div>
        <div>
          <div style="font-size:11px;font-weight:600;color:var(--pos);margin-bottom:8px;text-transform:uppercase;letter-spacing:.3px;">Where It's Used</div>
          ${NGRAM_USES.map(t => `<div style="display:flex;gap:6px;margin-bottom:8px;align-items:flex-start;"><div style="width:6px;height:6px;border-radius:50%;background:var(--pos);flex-shrink:0;margin-top:5px;"></div><div style="font-size:12px;color:#5a5950;line-height:1.55;">${escapeHtml(t)}</div></div>`).join("")}
        </div>
      </div>
      <div class="sv-insight i-bas">💡 You actually saw the biggest limitation — sparsity — happen live in the generator above, when the "of the" trigram had no data to work with.</div>
    </div>
  `;
}

// ─── SECTION: QUIZ ──────────────────────────────────────────────────────
function mountQuiz() {
  document.getElementById("secQuiz").innerHTML = `
    <div class="sv-eyebrow">Test Yourself</div>
    <div class="sv-card">
      <div class="sv-card-title">Concept Quiz</div>
      <p class="sv-desc">Pick an answer for instant feedback and a short explanation.</p>
      ${QUIZ.map((q, qi) => `
        ${qi > 0 ? '<hr class="qsep">' : ""}
        <div>
          <div style="display:flex;gap:8px;margin-bottom:10px;align-items:flex-start;">
            <span class="sv-badge b-bas" style="margin:2px 0 0;">${escapeHtml(q.tag)}</span>
            <div style="font-size:13.5px;font-weight:600;line-height:1.5;">${escapeHtml(q.q)}</div>
          </div>
          ${q.opts.map((opt, oi) => `<button class="qopt" id="ngqopt-${q.id}-${oi}" onclick="answerNgramQuiz(${q.id},${oi},${q.ans})">${escapeHtml(opt)}</button>`).join("")}
          <div class="sv-insight" id="ngqfb-${q.id}" style="display:none;"></div>
        </div>`).join("")}
      <div style="text-align:center;margin-top:20px;">
        <button class="sv-btn btn-sec" onclick="resetNgramQuiz()">↺ Reset quiz</button>
      </div>
    </div>
  `;
}

function answerNgramQuiz(qid, oi, ansIdx) {
  const firstBtn = document.getElementById(`ngqopt-${qid}-0`);
  if (firstBtn.disabled) return;
  const q = QUIZ.find(x => x.id === qid);
  q.opts.forEach((opt, i) => {
    const btn = document.getElementById(`ngqopt-${qid}-${i}`);
    btn.disabled = true;
    if (i === ansIdx) btn.classList.add("correct");
    else if (i === oi) btn.classList.add("incorrect");
  });
  const fb = document.getElementById(`ngqfb-${qid}`);
  const correct = oi === ansIdx;
  fb.style.display = "block";
  fb.className = `sv-insight ${correct ? "i-pos" : "i-warn"}`;
  fb.innerHTML = `<strong>${correct ? "✓ Correct!" : "✗ Incorrect."}</strong> ${escapeHtml(q.exp)}`;
}
window.answerNgramQuiz = answerNgramQuiz;

function resetNgramQuiz() { mountQuiz(); }
window.resetNgramQuiz = resetNgramQuiz;
