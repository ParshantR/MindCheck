/**
 * MindCheck — DASS-21 Frontend Application Logic
 * Communicates with Flask REST API at localhost:5000
 */

const API_BASE = "/api";

// ── State ─────────────────────────────────────────────────────────────────────
let questions      = [];
let answers        = {};   // { q01: 2, q02: 0, ... }
let demographics   = {};
let radarChart     = null;
let stressChart    = null;
let anxietyChart   = null;
let depressionChart = null;

// ── Severity helpers ──────────────────────────────────────────────────────────
const SEVERITY_COLORS = {
  "Normal":           "#10b981",
  "Mild":             "#f59e0b",
  "Moderate":         "#f97316",
  "Severe":           "#ef4444",
  "Extremely Severe": "#8b5cf6"
};
const SEVERITY_ICONS = {
  "stress":     "😓",
  "anxiety":    "😰",
  "depression": "😔"
};

// ── Entry point ───────────────────────────────────────────────────────────────
function startAssessment() {
  document.getElementById("section-demographics").classList.remove("hidden");
  document.getElementById("progress-bar-container").style.display = "block";
  document.getElementById("section-demographics").scrollIntoView({ behavior: "smooth", block: "start" });
  setStep(1);
}

// ── Step indicator ────────────────────────────────────────────────────────────
function setStep(step) {
  const steps  = [1, 2, 3];
  const fills  = { 1: "33%", 2: "66%", 3: "100%" };
  document.getElementById("progress-fill").style.width = fills[step];

  steps.forEach(s => {
    const el = document.getElementById(`step-${s}-indicator`);
    el.classList.remove("active", "done");
    if (s < step)  el.classList.add("done");
    if (s === step) el.classList.add("active");
  });
}

// ── STEP 1: Demographics validation ──────────────────────────────────────────
function submitDemographics() {
  const errorEl = document.getElementById("demo-error");
  errorEl.textContent = "";

  const age   = document.getElementById("age").value;
  const edu   = document.getElementById("education").value;
  const occ   = document.getElementById("occupation").value;
  const gender = document.querySelector('input[name="gender"]:checked');
  const marital = document.querySelector('input[name="marital"]:checked');
  const sleep  = document.querySelector('input[name="sleep"]:checked');

  if (!age || age < 10 || age > 100) {
    errorEl.textContent = "Please enter a valid age (10–100)."; return;
  }
  if (!gender)  { errorEl.textContent = "Please select your gender."; return; }
  if (!marital) { errorEl.textContent = "Please select your marital status."; return; }
  if (!edu)     { errorEl.textContent = "Please select your education level."; return; }
  if (!occ)     { errorEl.textContent = "Please select your occupation."; return; }
  if (!sleep)   { errorEl.textContent = "Please indicate if you have sleep problems."; return; }

  demographics = {
    age:            parseInt(age),
    gender:         parseInt(gender.value),
    marital_status: parseInt(marital.value),
    education:      parseInt(edu),
    occupation:     parseInt(occ),
    sleep_problem:  parseInt(sleep.value)
  };

  loadQuestions();
}

// ── STEP 2: Load & render questions ──────────────────────────────────────────
async function loadQuestions() {
  document.getElementById("section-demographics").classList.add("hidden");
  document.getElementById("section-questions").classList.remove("hidden");
  setStep(2);
  document.getElementById("section-questions").scrollIntoView({ behavior: "smooth" });

  try {
    const res  = await fetch(`${API_BASE}/questions?shuffle=true`);
    const data = await res.json();
    questions  = data.questions;
    renderQuestions(questions, data.response_options);
  } catch (err) {
    document.getElementById("questions-container").innerHTML =
      `<div class="error-msg">⚠️ Could not load questions. Please ensure the backend server is running on port 5000.</div>`;
  }
}

function renderQuestions(qs, options) {
  const container = document.getElementById("questions-container");
  container.innerHTML = "";

  qs.forEach((q, index) => {
    const item = document.createElement("div");
    item.className = "question-item";
    item.id = `question-${q.id}`;

    const optionsHTML = options.map(opt => `
      <button
        class="resp-btn"
        id="btn-${q.id}-${opt.value}"
        onclick="selectAnswer('${q.id}', ${opt.value})"
        title="${opt.description}"
      >
        <span class="resp-label">${opt.label}</span>
        <span class="resp-val">${opt.value}</span>
      </button>
    `).join("");

    item.innerHTML = `
      <div class="q-text">
        <span class="q-num-badge">Q${index + 1}</span>${q.text}
      </div>
      <div class="response-options" id="options-${q.id}">
        ${optionsHTML}
      </div>
    `;
    container.appendChild(item);
  });
  answers = {};
  updateCounter();
}

function selectAnswer(qid, value) {
  answers[qid] = value;

  // Update button styles
  for (let v = 0; v <= 3; v++) {
    const btn = document.getElementById(`btn-${qid}-${v}`);
    if (btn) btn.className = "resp-btn";
  }
  const selectedBtn = document.getElementById(`btn-${qid}-${value}`);
  if (selectedBtn) selectedBtn.className = `resp-btn selected-${value}`;

  updateCounter();
}

function updateCounter() {
  const answered = Object.keys(answers).length;
  const total    = questions.length;
  document.getElementById("q-counter").textContent = `${answered} / ${total} answered`;
}

function goBack() {
  document.getElementById("section-questions").classList.add("hidden");
  document.getElementById("section-demographics").classList.remove("hidden");
  setStep(1);
  document.getElementById("section-demographics").scrollIntoView({ behavior: "smooth" });
}

// ── STEP 3: Submit assessment ─────────────────────────────────────────────────
async function submitAssessment() {
  const errorEl = document.getElementById("q-error");
  errorEl.textContent = "";

  const unanswered = questions.filter(q => !(q.id in answers));
  if (unanswered.length > 0) {
    errorEl.textContent = `Please answer all questions. ${unanswered.length} question(s) remaining.`;
    // Scroll to first unanswered
    const firstUnanswered = document.getElementById(`question-${unanswered[0].id}`);
    if (firstUnanswered) firstUnanswered.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  // Disable submit button + show loading
  const submitBtn = document.getElementById("submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Analysing...";

  try {
    const payload = {
      demographics: demographics,
      responses:    answers
    };

    const res  = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      errorEl.textContent = `Error: ${data.error || "Unknown error occurred."}`;
      submitBtn.disabled = false;
      submitBtn.textContent = "Get My Results →";
      return;
    }

    // Re-enable button BEFORE hiding the section — so retake works
    submitBtn.disabled = false;
    submitBtn.textContent = "Get My Results →";

    renderResults(data.data);

  } catch (err) {
    errorEl.textContent = "Could not connect to server. Please ensure the backend is running.";
    submitBtn.disabled = false;
    submitBtn.textContent = "Get My Results →";
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById("section-questions").classList.add("hidden");
  document.getElementById("section-results").classList.remove("hidden");
  setStep(3);
  document.getElementById("section-results").scrollIntoView({ behavior: "smooth" });

  const predictions = data.predictions;

  // ── Overall Risk ────────────────────────────────────────────────────────────
  const overallScore = data.overall_risk_score;
  document.getElementById("overall-risk-score").textContent = overallScore.toFixed(1) + " / 5";
  document.getElementById("overall-risk-desc").textContent = getOverallDescription(overallScore);

  // ── Condition Cards ─────────────────────────────────────────────────────────
  const cardsContainer = document.getElementById("condition-cards");
  cardsContainer.innerHTML = "";
  const conditionOrder = ["stress", "anxiety", "depression"];

  conditionOrder.forEach(cond => {
    const pred = predictions[cond];
    const sev  = pred.severity_level;
    const card = document.createElement("div");
    card.className = `condition-card sev-${sev}`;
    card.innerHTML = `
      <div class="cond-icon">${SEVERITY_ICONS[cond]}</div>
      <div class="cond-name">${cond}</div>
      <div class="cond-label">${pred.severity_label}</div>
      <div class="cond-score">Score: ${pred.raw_score}/21</div>
      <div class="cond-conf">Confidence: ${(pred.confidence * 100).toFixed(0)}%</div>
    `;
    cardsContainer.appendChild(card);
  });

  // ── Radar Chart ─────────────────────────────────────────────────────────────
  destroyChart(radarChart);
  const radarCtx = document.getElementById("radarChart").getContext("2d");
  radarChart = new Chart(radarCtx, {
    type: "radar",
    data: {
      labels: ["Stress", "Anxiety", "Depression"],
      datasets: [{
        label: "Your Severity (1-5)",
        data: conditionOrder.map(c => predictions[c].severity_level),
        backgroundColor: "rgba(79,70,229,0.2)",
        borderColor:     "rgba(79,70,229,0.8)",
        borderWidth: 2,
        pointBackgroundColor: conditionOrder.map(c =>
          SEVERITY_COLORS[predictions[c].severity_label] || "#4f46e5"
        ),
        pointRadius: 6
      }]
    },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0, max: 5,
          ticks: { stepSize: 1, font: { size: 11 } },
          pointLabels: { font: { size: 13, weight: "bold" } }
        }
      },
      plugins: { legend: { display: false } }
    }
  });

  // ── Probability Distribution Charts ─────────────────────────────────────────
const levels = ["Normal","Mild","Moderate","Severe","Extremely Severe"];
const probColors = ["#10b981","#f59e0b","#f97316","#ef4444","#8b5cf6"];

// Extract probabilities EXPLICITLY by key — never use Object.values()
// because JSON key order is not guaranteed across all environments
function extractProbs(probObj) {
  return levels.map(level => probObj[level] || 0);
}

destroyChart(stressChart);
stressChart = makeProbChart("stressChart",
  extractProbs(predictions.stress.probabilities), levels, probColors
);

destroyChart(anxietyChart);
anxietyChart = makeProbChart("anxietyChart",
  extractProbs(predictions.anxiety.probabilities), levels, probColors
);

destroyChart(depressionChart);
depressionChart = makeProbChart("depressionChart",
  extractProbs(predictions.depression.probabilities), levels, probColors
);

  // ── Comorbidity Index ────────────────────────────────────────────────────────
  const ci = data.comorbidity_index;
  const comorbContainer = document.getElementById("comorbidity-display");
  const dots = [0,1,2].map(i =>
    `<div class="comorbidity-dot ${i < ci ? 'active' : ''}"></div>`
  ).join("");
  const ciDesc = ci === 0 ? "No conditions are currently elevated." :
                 ci === 1 ? "One condition is elevated above the clinical threshold." :
                 ci === 2 ? "Two conditions are simultaneously elevated — please monitor carefully." :
                            "All three conditions are simultaneously elevated — please seek professional support.";

  comorbContainer.innerHTML = `
    <div class="comorbidity-score-big">${ci}</div>
    <div class="comorbidity-dots">${dots}</div>
    <div class="comorbidity-text">&nbsp;${ciDesc}</div>
  `;

  // ── Feedback per condition ───────────────────────────────────────────────────
  const feedbackContainer = document.getElementById("feedback-sections");
  feedbackContainer.innerHTML = "<h3 style='margin-bottom:16px;font-size:1.2rem;font-weight:700;'>Recommendations</h3>";

  conditionOrder.forEach(cond => {
    const pred     = predictions[cond];
    const feedback = pred.feedback;
    const sev      = pred.severity_level;

    const warningHTML = feedback.warning ? `
      <div class="feedback-warning">⚠️ ${feedback.warning}</div>
    ` : "";

    const suggestionsHTML = feedback.suggestions.map(s =>
      `<li>${s}</li>`
    ).join("");

    const card = document.createElement("div");
    card.className = `feedback-card sev-${sev}`;
    card.innerHTML = `
      <h4>${SEVERITY_ICONS[cond]} ${cond.charAt(0).toUpperCase()+cond.slice(1)} — ${pred.severity_label}</h4>
      <div class="feedback-summary">${feedback.summary}</div>
      ${warningHTML}
      <ul class="suggestions-list">${suggestionsHTML}</ul>
    `;
    feedbackContainer.appendChild(card);
  });
}

// ── Chart helpers ─────────────────────────────────────────────────────────────
function makeProbChart(canvasId, values, labels, colors) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + "cc"),
        borderColor:     colors,
        borderWidth: 2,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${(ctx.raw * 100).toFixed(1)}%`
          }
        }
      },
      scales: {
        y: {
          min: 0, max: 1,
          ticks: { callback: v => `${(v*100).toFixed(0)}%`, font: { size: 10 } }
        },
        x: { ticks: { font: { size: 9 } } }
      }
    }
  });
}

function destroyChart(chart) {
  if (chart) { try { chart.destroy(); } catch(e) {} }
}

// ── Overall Risk Description ──────────────────────────────────────────────────
function getOverallDescription(score) {
  if (score <= 1.5) return "Your overall mental health appears to be in a good range.";
  if (score <= 2.5) return "You are experiencing mild levels across conditions. Monitor and manage.";
  if (score <= 3.5) return "Moderate overall risk — consider speaking with a professional.";
  if (score <= 4.5) return "High overall risk — professional support is strongly recommended.";
  return "Very high overall risk — please seek professional help as soon as possible.";
}

// ── Retake ────────────────────────────────────────────────────────────────────
function retakeAssessment() {
  // Destroy all charts so they don't overlap on next submission
  destroyChart(radarChart);
  destroyChart(stressChart);
  destroyChart(anxietyChart);
  destroyChart(depressionChart);
  radarChart      = null;
  stressChart     = null;
  anxietyChart    = null;
  depressionChart = null;

  // Reset all state
  answers      = {};
  questions    = [];
  demographics = {};

  // Reset submit button in case user is retaking
  const submitBtn = document.getElementById("submit-btn");
  if (submitBtn) {
    submitBtn.disabled = false;
    submitBtn.textContent = "Get My Results →";
  }

  // Clear any error messages
  document.getElementById("demo-error").textContent = "";
  const qError = document.getElementById("q-error");
  if (qError) qError.textContent = "";

  // Navigate back to step 1
  document.getElementById("section-results").classList.add("hidden");
  document.getElementById("section-questions").classList.add("hidden");
  document.getElementById("section-demographics").classList.remove("hidden");
  setStep(1);
  window.scrollTo({ top: 0, behavior: "smooth" });
}
