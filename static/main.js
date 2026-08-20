// PyStereo web UI - client logic

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// -- DOM refs ---------------------------------------------------------------

const modelBanner = $("#modelBanner");
const modelBannerMsg = $("#modelBannerMessage");
const modelProgressWrap = $("#modelBannerProgressWrap");
const modelProgressBar = $("#modelBannerProgressBar");
const modelProgressLabel = $("#modelBannerProgressLabel");
const btnModelDownload = $("#btnModelDownload");
const btnModelCancel = $("#btnModelCancel");
const btnModelRemove = $("#btnModelRemove");
const modelGateOverlay = $("#modelGateOverlay");
const gateHint = $(".model-gate-hint");

const dropZone = $("#dropZone");
const fileInput = $("#fileInput");
const previewImg = $("#previewImg");
const methodSelect = $("#methodSelect");
const maxDimSlider = $("#maxDimSlider");
const maxDimValue = $("#maxDimValue");
const btnGenerate = $("#btnGenerate");
const statusWrap = $("#statusWrap");
const statusBar = $("#statusBar");

const pipelinePlaceholder = $("#pipelinePlaceholder");
const pipelineStages = $("#pipelineStages");
const stageOriginal = $("#stageOriginal");
const stageOriginalImg = $("#stageOriginalImg");
const stageDepth = $("#stageDepth");
const stageDepthImg = $("#stageDepthImg");
const stageSbs = $("#stageSbs");
const stageSbsImg = $("#stageSbsImg");
const stageTiming = $("#stageTiming");

// -- State ------------------------------------------------------------------

let selectedFile = null;
let modelReady = false;
let generating = false;
let pollTimer = null;

// -- Formatting helpers -----------------------------------------------------

function fmtBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  if (bytes < 1024 * 1024 * 1024)
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

// -- Model status -----------------------------------------------------------

function applyModelStatus(data) {
  if (!data) return;

  modelBanner.hidden = false;
  modelBanner.classList.remove("is-ready", "is-error", "is-downloading");

  const state = data.state || "idle";
  modelReady = state === "ready";

  if (state === "ready") {
    modelBanner.classList.add("is-ready");
    modelBannerMsg.textContent = "AI stereo models ready";
    modelProgressWrap.classList.add("hidden");
    btnModelDownload.hidden = true;
    btnModelCancel.hidden = true;
    btnModelRemove.hidden = false;
    unlockGate();
  } else if (state === "downloading") {
    modelBanner.classList.add("is-downloading");
    modelBannerMsg.textContent = data.message || "Downloading AI models...";
    modelProgressWrap.classList.remove("hidden");
    const pct = data.percent || 0;
    modelProgressBar.style.width = pct + "%";
    const done = data.bytes_downloaded || 0;
    const total = data.bytes_total || 0;
    modelProgressLabel.textContent =
      total > 0 ? fmtBytes(done) + " / " + fmtBytes(total) + " (" + pct + "%)" : pct + "%";
    btnModelDownload.hidden = true;
    btnModelCancel.hidden = false;
    btnModelCancel.disabled = false;
    btnModelRemove.hidden = true;
    lockGate("Download in progress - workspace unlocks when it finishes.");
  } else if (state === "error") {
    modelBanner.classList.add("is-error");
    modelBannerMsg.textContent = data.error
      ? "Download failed: " + data.error
      : "Download failed. Try again.";
    modelProgressWrap.classList.add("hidden");
    btnModelDownload.hidden = false;
    btnModelDownload.textContent = "Retry";
    btnModelDownload.disabled = false;
    btnModelCancel.hidden = true;
    btnModelRemove.hidden = true;
    lockGate("Download failed. Use Retry in the banner above.");
  } else {
    modelBannerMsg.textContent =
      "Download the AI stereo models to generate side-by-side images.";
    modelProgressWrap.classList.add("hidden");
    btnModelDownload.hidden = false;
    btnModelDownload.textContent = "Download";
    btnModelDownload.disabled = false;
    btnModelCancel.hidden = true;
    btnModelRemove.hidden = true;
    lockGate("Download AI models using the banner above to get started.");
  }

  updateGenerateBtn();
}

function lockGate(hint) {
  document.body.classList.add("model-gate-active");
  modelGateOverlay.hidden = false;
  modelGateOverlay.setAttribute("aria-hidden", "false");
  if (gateHint && hint) gateHint.textContent = hint;
}

function unlockGate() {
  document.body.classList.remove("model-gate-active");
  modelGateOverlay.hidden = true;
  modelGateOverlay.setAttribute("aria-hidden", "true");
}

async function pollModelStatus() {
  try {
    const res = await fetch("/api/model/status");
    if (!res.ok) return;
    const data = await res.json();
    applyModelStatus(data);

    if (data.state === "downloading") {
      schedulePoll(1500);
    } else {
      schedulePoll(10000);
    }
  } catch {
    schedulePoll(5000);
  }
}

function schedulePoll(ms) {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(pollModelStatus, ms);
}

// -- Model actions ----------------------------------------------------------

btnModelDownload.addEventListener("click", async () => {
  btnModelDownload.disabled = true;
  try {
    const res = await fetch("/api/model/download", { method: "POST" });
    const data = await res.json();
    applyModelStatus(data);
    schedulePoll(1500);
  } catch {
    btnModelDownload.disabled = false;
  }
});

btnModelCancel.addEventListener("click", async () => {
  btnModelCancel.disabled = true;
  try {
    const res = await fetch("/api/model/cancel", { method: "POST" });
    const data = await res.json();
    applyModelStatus(data);
  } catch {
    btnModelCancel.disabled = false;
  }
});

btnModelRemove.addEventListener("click", async () => {
  if (!confirm("Delete downloaded model weights? You can re-download later.")) return;
  btnModelRemove.disabled = true;
  try {
    const res = await fetch("/api/model/delete", { method: "POST" });
    const data = await res.json();
    applyModelStatus(data);
  } catch {
    btnModelRemove.disabled = false;
  }
});

// -- File input -------------------------------------------------------------

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) selectFile(files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  if (!file.type.startsWith("image/") && !file.name.match(/\.(heic|heif)$/i)) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.remove("hidden");
  dropZone.classList.add("has-preview");
  updateGenerateBtn();
}

// -- Settings ---------------------------------------------------------------

maxDimSlider.addEventListener("input", () => {
  maxDimValue.textContent = maxDimSlider.value;
});

async function loadMethods() {
  try {
    const res = await fetch("/api/stereo-methods");
    if (!res.ok) return;
    const methods = await res.json();
    for (const m of methods) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m.replace(/_/g, " ");
      methodSelect.appendChild(opt);
    }
  } catch {
    // Methods will use default
  }
}

// -- Generate ---------------------------------------------------------------

function updateGenerateBtn() {
  btnGenerate.disabled = !selectedFile || !modelReady || generating;
}

function setStatus(text, busy, isError) {
  statusBar.textContent = text;
  statusBar.className = "status-bar" + (isError ? " error" : busy ? " working" : "");
  statusWrap.classList.toggle("is-busy", !!busy);
  statusWrap.setAttribute("aria-busy", busy ? "true" : "false");
}

btnGenerate.addEventListener("click", async () => {
  if (!selectedFile || !modelReady || generating) return;
  generating = true;
  updateGenerateBtn();

  // Reset output
  pipelinePlaceholder.classList.add("hidden");
  pipelineStages.classList.remove("hidden");
  stageOriginal.hidden = false;
  stageOriginalImg.src = URL.createObjectURL(selectedFile);
  stageDepth.hidden = true;
  stageSbs.hidden = true;
  stageTiming.textContent = "";

  setStatus("Running depth estimation + stereo synthesis...", true, false);

  const form = new FormData();
  form.append("file", selectedFile);
  const method = methodSelect.value;
  if (method) form.append("method", method);
  form.append("max_dim", maxDimSlider.value);

  try {
    const res = await fetch("/api/generate", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      setStatus(data.error || "Generation failed", false, true);
      generating = false;
      updateGenerateBtn();
      return;
    }

    // Show depth
    if (data.depth_url) {
      stageDepthImg.src = data.depth_url;
      stageDepth.hidden = false;
    }

    // Show SBS
    if (data.sbs_url) {
      stageSbsImg.src = data.sbs_url;
      stageSbs.hidden = false;
    }

    const elapsed = data.elapsed_seconds != null ? data.elapsed_seconds.toFixed(1) : "?";
    const dims = data.width && data.height ? data.width + "x" + data.height : "";
    const methodUsed = data.method || "default";
    stageTiming.textContent =
      "Method: " + methodUsed + " - " + dims + " - " + elapsed + "s";

    setStatus("Done (" + elapsed + "s)", false, false);
  } catch (err) {
    setStatus("Request failed: " + err.message, false, true);
  }

  generating = false;
  updateGenerateBtn();
});

// -- Init -------------------------------------------------------------------

loadMethods();
pollModelStatus();
