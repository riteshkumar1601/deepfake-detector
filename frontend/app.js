// ============================================================
// Deepfake Detector — Frontend logic
// Talks to the FastAPI backend at http://localhost:8000
//
// Preferred: open http://localhost:8000 directly (the backend serves this
// page too). If the page is opened from disk (file://) or from another dev
// server, API calls fall back to http://localhost:8000.
// ============================================================

const API_BASE =
  window.location.port === "8000" ? window.location.origin : "http://localhost:8000";

// ----- State -----
let selectedFiles = [];
let currentResults = [];
let currentReportId = null;
let currentPdfAvailable = false;
let isAnalyzing = false;
let loaderTimer = null;
let previewUrls = [];

// ----- Element references -----
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const previewGrid = document.getElementById("previewGrid");
const uploadActions = document.getElementById("uploadActions");
const fileCount = document.getElementById("fileCount");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const uploadPanel = document.getElementById("uploadPanel");
const loaderPanel = document.getElementById("loaderPanel");
const resultsPanel = document.getElementById("resultsPanel");
const loaderProgressBar = document.getElementById("loaderProgressBar");
const loaderTitle = document.getElementById("loaderTitle");

const summaryCards = document.getElementById("summaryCards");
const resultsTableBody = document.getElementById("resultsTableBody");
const resultsSubtitle = document.getElementById("resultsSubtitle");
const downloadBtn = document.getElementById("downloadBtn");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

const modalOverlay = document.getElementById("modalOverlay");
const modalClose = document.getElementById("modalClose");
const modalFilename = document.getElementById("modalFilename");
const modalVerdict = document.getElementById("modalVerdict");
const modalPanelImg = document.getElementById("modalPanelImg");
const modalGaugeImg = document.getElementById("modalGaugeImg");
const modalShapImg = document.getElementById("modalShapImg");

const toast = document.getElementById("toast");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// ============================================================
// SERVER HEALTH CHECK — runs on page load
// ============================================================
async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      statusDot.className = "status-dot online";
      statusText.textContent = "Server online";
      maybeRedirectToBackend();
    } else {
      throw new Error();
    }
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Server offline — start server.py";
  }
}
checkServerHealth();

// ============================================================
// SAME-ORIGIN REDIRECT — avoids the "Reload site?" loop
// ============================================================
// If this page was opened through file:// or a dev server (e.g. VS Code
// Live Server on port 5500), that server reloads the page every time a file
// in the workspace changes. The backend writes uploads/visuals/reports while
// analyzing, so the page used to reload mid-test, pop the "Reload site?"
// dialog and wipe all results.
//
// The backend also serves this exact page at http://localhost:8000 — there
// nothing watches the filesystem, so we move there once, automatically.
// (Open the page with ?stay=1 if you intentionally want to stay on a dev
// server, e.g. while editing CSS.)
function maybeRedirectToBackend() {
  if (window.location.port === "8000") return; // already served by the backend

  let skip = false;
  try {
    if (new URLSearchParams(window.location.search).get("stay") === "1") {
      sessionStorage.setItem("dfd_stay", "1");
    }
    skip = sessionStorage.getItem("dfd_stay") === "1" ||
           sessionStorage.getItem("dfd_redirected") === "1";
  } catch { /* storage unavailable (private mode) — redirect anyway */ }
  if (skip) return;

  // Only redirect if the backend really serves the frontend at :8000.
  fetch(`${API_BASE}/`, { method: "HEAD" })
    .then((res) => {
      if (!res.ok) throw new Error("backend is not serving the frontend");
      try { sessionStorage.setItem("dfd_redirected", "1"); } catch { /* ignore */ }
      window.location.replace(`${API_BASE}/`);
    })
    .catch(() => { /* stay here; the app still works cross-origin via CORS */ });
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================
function showToast(message, type = "default") {
  toast.textContent = message;
  toast.className = `toast show ${type}`;
  setTimeout(() => { toast.className = "toast"; }, 3500);
}

// ============================================================
// FILE SELECTION — drag & drop + browse
// ============================================================
browseBtn.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", (e) => {
  if (e.target === browseBtn) return;
  fileInput.click();
});

fileInput.addEventListener("change", (e) => addFiles(Array.from(e.target.files)));

["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
  addFiles(files);
});

function addFiles(files) {
  if (!files.length) return;
  selectedFiles = selectedFiles.concat(files);
  renderPreviews();
}

function renderPreviews() {
  // Release object URLs from the previous render so the browser doesn't hold
  // on to large image blobs (this grows fast when several images are picked).
  previewUrls.forEach(u => URL.revokeObjectURL(u));
  previewUrls = [];

  previewGrid.innerHTML = "";
  selectedFiles.forEach((file, idx) => {
    const url = URL.createObjectURL(file);
    previewUrls.push(url);
    const item = document.createElement("div");
    item.className = "preview-item";
    item.innerHTML = `
      <img src="${url}" alt="${file.name}">
      <button class="preview-remove" data-idx="${idx}" title="Remove">✕</button>
    `;
    previewGrid.appendChild(item);
  });

  previewGrid.querySelectorAll(".preview-remove").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx);
      selectedFiles.splice(idx, 1);
      renderPreviews();
    });
  });

  if (selectedFiles.length > 0) {
    uploadActions.style.display = "flex";
    fileCount.textContent = `${selectedFiles.length} image${selectedFiles.length > 1 ? "s" : ""} selected`;
  } else {
    uploadActions.style.display = "none";
  }
}

clearBtn.addEventListener("click", () => {
  selectedFiles = [];
  fileInput.value = "";
  renderPreviews();
});

// ============================================================
// ANALYZE — send to backend, show loader, render results
// ============================================================
const analyzeBtnLabel = analyzeBtn.querySelector("span");

analyzeBtn.addEventListener("click", async () => {
  // Guard against double clicks: a second request would run while the first
  // is still processing and would overwrite the page state mid-flight.
  if (selectedFiles.length === 0 || isAnalyzing) return;
  isAnalyzing = true;
  analyzeBtn.disabled = true;
  if (analyzeBtnLabel) analyzeBtnLabel.textContent = "Analyzing…";

  uploadPanel.style.display = "none";
  resultsPanel.style.display = "none";
  loaderPanel.style.display = "block";
  animateLoaderProgress();

  const formData = new FormData();
  selectedFiles.forEach(file => formData.append("files", file));

  // Safety net: never spin forever if the connection dies silently.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15 * 60 * 1000);

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || `Server responded with status ${res.status}`);
    }

    const data = await res.json();
    if (!data || !Array.isArray(data.results) || !data.summary) {
      throw new Error("Backend returned an invalid response payload.");
    }

    currentResults = data.results;
    currentReportId = data.report_id;
    currentPdfAvailable = Boolean(data.download_url);

    clearLoaderProgress();
    loaderProgressBar.style.width = "100%";
    loaderPanel.style.display = "none";
    renderResults(data);
    resultsPanel.style.display = "block";
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast(`✅ Analysis complete — ${data.summary.total} image(s) processed`, "success");

  } catch (err) {
    clearLoaderProgress();
    loaderPanel.style.display = "none";
    uploadPanel.style.display = "block";
    const msg = err.name === "AbortError"
      ? "the server took too long to respond."
      : err.message;
    showToast(`❌ Analysis failed: ${msg}`, "error");
    console.error(err);
  } finally {
    clearTimeout(timeoutId);
    isAnalyzing = false;
    analyzeBtn.disabled = false;
    if (analyzeBtnLabel) analyzeBtnLabel.textContent = "Start Testing";
  }
});

function clearLoaderProgress() {
  if (loaderTimer) {
    clearInterval(loaderTimer);
    loaderTimer = null;
  }
}

function animateLoaderProgress() {
  clearLoaderProgress();
  let pct = 0;
  loaderProgressBar.style.width = "0%";
  loaderTimer = setInterval(() => {
    pct += Math.random() * 12;
    if (pct >= 92) { clearInterval(loaderTimer); loaderTimer = null; pct = 92; }
    loaderProgressBar.style.width = `${pct}%`;
    loaderTitle.textContent = pct < 50 ? "Extracting features…" : "Running model ensemble…";
  }, 350);
}

// ============================================================
// RENDER RESULTS
// ============================================================
function renderResults(data) {
  // Defensive: never let one odd/missing field crash the whole results view.
  const summary = data.summary || {};
  const results = Array.isArray(data.results) ? data.results : [];
  const counts = {
    total: summary.total != null ? summary.total : results.length,
    ai_generated: summary.ai_generated || 0,
    real: summary.real || 0,
    errors: summary.errors || 0,
  };

  resultsSubtitle.textContent =
    `${counts.total} image(s) analyzed — ${counts.ai_generated} AI-generated, ${counts.real} real` +
    (counts.errors ? `, ${counts.errors} failed` : "");

  summaryCards.innerHTML = `
    <div class="summary-card">
      <div class="num">${counts.total}</div>
      <div class="label">Total Analyzed</div>
    </div>
    <div class="summary-card ai">
      <div class="num">${counts.ai_generated}</div>
      <div class="label">AI-Generated</div>
    </div>
    <div class="summary-card real">
      <div class="num">${counts.real}</div>
      <div class="label">Real</div>
    </div>
  `;

  resultsTableBody.innerHTML = "";
  results.forEach((r, idx) => {
    const row = document.createElement("tr");

    if (r.verdict === "ERROR") {
      row.innerHTML = `
        <td colspan="7" style="color:#A4331F;">⚠️ ${r.filename} — failed to process (${r.error || "unknown error"})</td>
      `;
      resultsTableBody.appendChild(row);
      return;
    }

    const riskScore = Number(r.risk_score) || 0;
    const confidence = r.confidence_pct != null ? r.confidence_pct : "—";
    const verdictClass = r.verdict === "AI-GENERATED" ? "ai" : "real";
    const riskColor = riskScore >= 70 ? "#A4331F" : riskScore >= 40 ? "#E8A33D" : "#2E7D5B";
    const visuals = r.visuals || {};
    const thumbUrl = visuals.panel ? `${API_BASE}${visuals.panel}` : "";

    row.innerHTML = `
      <td class="thumb-cell"><img src="${thumbUrl}" alt="thumb" onerror="this.style.display='none'"></td>
      <td>${r.filename}</td>
      <td><span class="verdict-badge ${verdictClass}">${r.verdict}</span></td>
      <td>${confidence}%</td>
      <td>
        <div class="risk-bar-wrap">
          <div class="risk-bar-track"><div class="risk-bar-fill" style="width:${riskScore}%; background:${riskColor};"></div></div>
          <span class="risk-num">${riskScore}</span>
        </div>
      </td>
      <td>${r.face_detected ? "Yes" : "No"}</td>
      <td><button class="details-link" data-idx="${idx}">View Details</button></td>
    `;
    resultsTableBody.appendChild(row);
  });

  resultsTableBody.querySelectorAll(".details-link").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.currentTarget.dataset.idx, 10);
      const result = results[idx];
      if (result) openModal(result);
    });
  });
}

// ============================================================
// DETAIL MODAL
// ============================================================
function openModal(result) {
  const visuals = result.visuals || {};
  modalFilename.textContent = result.filename || "—";
  modalVerdict.textContent = `${result.verdict} (${result.confidence_pct}% confidence)`;
  modalVerdict.className = `modal-verdict ${result.verdict === "AI-GENERATED" ? "ai" : "real"}`;

  [["panel", modalPanelImg], ["gauge", modalGaugeImg], ["shap", modalShapImg]].forEach(([key, el]) => {
    if (visuals[key]) {
      el.src = `${API_BASE}${visuals[key]}`;
      el.style.display = "";
    } else {
      el.removeAttribute("src");
      el.style.display = "none";
    }
  });

  modalOverlay.style.display = "flex";
}
modalClose.addEventListener("click", () => modalOverlay.style.display = "none");
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) modalOverlay.style.display = "none";
});

// ============================================================
// DOWNLOAD REPORT
// ============================================================
downloadBtn.addEventListener("click", () => {
  if (!currentReportId) {
    showToast("⚠️ No report available yet — run an analysis first.", "error");
    return;
  }
  if (!currentPdfAvailable) {
    showToast("⚠️ PDF report is not available (no image was processed successfully).", "error");
    return;
  }

  const url = `${API_BASE}/api/download-report/${currentReportId}`;

  // Same-origin (page served by the backend): use a direct download link —
  // no popup blocker can interfere. Cross-origin: open in a new tab; the
  // response carries Content-Disposition: attachment, so it downloads.
  if (url.startsWith(window.location.origin)) {
    const a = document.createElement("a");
    a.href = url;
    a.download = `Deepfake_Detection_Report_${currentReportId}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } else {
    window.open(url, "_blank");
  }
  showToast("📄 Downloading PDF report…");
});

// ============================================================
// NEW ANALYSIS — reset everything
// ============================================================
newAnalysisBtn.addEventListener("click", () => {
  selectedFiles = [];
  currentResults = [];
  currentReportId = null;
  currentPdfAvailable = false;
  fileInput.value = "";
  renderPreviews();
  resultsPanel.style.display = "none";
  uploadPanel.style.display = "block";
});

// ============================================================
// RELOAD GUARD — if something (e.g. a dev-server file watcher) tries to
// reload the page mid-analysis, ask the user before discarding the work.
// ============================================================
window.addEventListener("beforeunload", (e) => {
  if (isAnalyzing) {
    e.preventDefault();
    e.returnValue = "";
  }
});