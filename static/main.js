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
const depthModelRow = $("#depthModelRow");
const depthModelSelect = $("#depthModelSelect");
const btnDownloadDepth = $("#btnDownloadDepth");
const methodSelect = $("#methodSelect");
const methodInfo = $("#methodInfo");
const taichiStatus = $("#taichiStatus");
const sharpBanner = $("#sharpBanner");
const sharpBannerText = $(".sharp-banner-text");
const btnDownloadSharp = $("#btnDownloadSharp");
const btnCancelSharp = $("#btnCancelSharp");
const btnRemoveSharp = $("#btnRemoveSharp");
const maxDimRow = $("#maxDimRow");
const maxDimSlider = $("#maxDimSlider");
const maxDimValue = $("#maxDimValue");
const disableCacheCheck = $("#disableCacheCheck");
const cacheSharpValue = $("#cacheSharpValue");
const cacheOutputsValue = $("#cacheOutputsValue");
const loadedModelsValue = $("#loadedModelsValue");
const btnClearSharp = $("#btnClearSharp");
const btnClearOutputs = $("#btnClearOutputs");
const btnUnloadModels = $("#btnUnloadModels");
const sharpCacheMaxInput = $("#sharpCacheMaxInput");
const outputsKeepInput = $("#outputsKeepInput");
const sharpIdleInput = $("#sharpIdleInput");
const cacheSharpPath = $("#cacheSharpPath");
const cacheOutputsPath = $("#cacheOutputsPath");
let lastCacheStats = null;
const btnGenerate = $("#btnGenerate");
const statusWrap = $("#statusWrap");
const statusBar = $("#statusBar");

const pipelinePlaceholder = $("#pipelinePlaceholder");
const pipelineStages = $("#pipelineStages");
const stageOriginal = $("#stageOriginal");
const stageOriginalImg = $("#stageOriginalImg");
const stageSplat = $("#stageSplat");
const stageSplatImg = $("#stageSplatImg");
const stageDepth = $("#stageDepth");
const stageDepthImg = $("#stageDepthImg");
const stageWarp = $("#stageWarp");
const stageWarpImg = $("#stageWarpImg");
const stageSbs = $("#stageSbs");
const stageSbsImg = $("#stageSbsImg");
const stageTiming = $("#stageTiming");
const stageSteps = $("#stageSteps");

const lightbox = $("#lightbox");
const lightboxImg = $("#lightboxImg");
const lightboxCaption = $("#lightboxCaption");
const lightboxClose = $("#lightboxClose");
const lightboxPrev = $("#lightboxPrev");
const lightboxNext = $("#lightboxNext");

// -- State ------------------------------------------------------------------

let selectedFile = null;
let previewObjectUrl = null;
let stageOriginalObjectUrl = null;
let modelReady = false;
let sharpReady = false;
let sharpDownloading = false;
/** SHARP download is waiting for the base model download to finish. */
let sharpQueued = false;
let sharpPercent = 0;
let modelDownloading = false;
/** Ignore stale sharp_downloading:false until this timestamp (ms). */
let sharpDownloadGraceUntil = 0;
let generating = false;
let pollTimer = null;
let depthModelStatus = {};
let methodMeta = {};
let defaultMethodName = "per_eye_inpaint";
let taichiRenderAvailable = false;

function methodOptionLabel(m) {
  let label = m.label;
  if (m.default) label += " (default)";
  if (m.deprecated) label += " (deprecated)";
  return label;
}

function taichiRenderNote(meta) {
  if (!meta || !meta.uses_taichi) return "";
  if (taichiRenderAvailable) {
    return "Render: Taichi (Metal/GPU).";
  }
  return "Render: torch fallback in this build (same output, slower splat step).";
}

function refreshMethodOptionLabels() {
  for (const opt of methodSelect.options) {
    const m = methodMeta[opt.value];
    if (!m) continue;
    opt.textContent = methodOptionLabel(m);
  }
}

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
  modelDownloading = state === "downloading";
  applySharpStatus(data);

  if (state === "ready") {
    modelBanner.classList.add("is-ready");
    modelBannerMsg.textContent = "AI stereo models ready";
    modelProgressWrap.classList.add("hidden");
    btnModelDownload.hidden = true;
    btnModelCancel.hidden = true;
    btnModelRemove.hidden = false;
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
  } else {
    modelBannerMsg.textContent =
      "Download the AI stereo models to generate side-by-side images.";
    modelProgressWrap.classList.add("hidden");
    btnModelDownload.hidden = false;
    btnModelDownload.textContent = "Download";
    btnModelDownload.disabled = false;
    btnModelCancel.hidden = true;
    btnModelRemove.hidden = true;
  }

  syncModelGate(state);
  updateSharpBanner();
  updateGenerateBtn();
}

/** Merge server SHARP fields without letting a stale poll undo a just-started download. */
function applySharpStatus(data) {
  if (!data) return;
  if (data.sharp_ready) {
    sharpReady = true;
    sharpDownloading = false;
    sharpQueued = false;
    sharpPercent = 100;
    sharpDownloadGraceUntil = 0;
    return;
  }
  sharpReady = false;
  if (data.sharp_downloading) {
    sharpDownloading = true;
    sharpQueued = !!data.sharp_queued;
    sharpPercent = data.sharp_percent || 0;
    sharpDownloadGraceUntil = Date.now() + 8000;
    return;
  }
  if (Date.now() < sharpDownloadGraceUntil) {
    // Keep optimistic / in-flight downloading UI; wait for the next poll.
    return;
  }
  sharpDownloading = false;
  sharpQueued = false;
  sharpPercent = 0;
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

/** Unlock for SHARP-only methods; lock depth methods until the pack is ready. */
function syncModelGate(state) {
  if (!selectedMethodNeedsDepth() || modelReady) {
    unlockGate();
    return;
  }
  if (state === "downloading") {
    lockGate("Download in progress - workspace unlocks when it finishes.");
  } else if (state === "error") {
    lockGate("Download failed. Use Retry in the banner above.");
  } else {
    lockGate("Download AI models using the banner above to get started.");
  }
}

async function pollModelStatus() {
  try {
    const res = await fetch("/api/model/status");
    if (!res.ok) return;
    const data = await res.json();
    applyModelStatus(data);

    if (data.state === "downloading" || data.sharp_downloading || sharpDownloading) {
      schedulePoll(1500);
    } else if (selectedMethodNeedsSharp() && !sharpReady) {
      // SHARP still missing - poll often so a background download is noticed.
      schedulePoll(2000);
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
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      applyModelStatus({ state: "error", error: msg });
      return;
    }
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
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      applyModelStatus({ state: "error", error: msg });
      return;
    }
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
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      applyModelStatus({ state: "error", error: msg });
      return;
    }
    const data = await res.json();
    applyModelStatus(data);
  } catch {
    btnModelRemove.disabled = false;
  }
});

// -- Depth model selector ---------------------------------------------------

async function loadDepthModels() {
  try {
    const res = await fetch("/api/depth-models");
    if (!res.ok) {
      console.error("depth-models failed", res.status, await res.text());
      return;
    }
    const models = await res.json();
    for (const m of models) {
      depthModelStatus[m.size] = m.downloaded;
    }
    updateDepthDownloadBtn();
  } catch (err) {
    console.error("depth-models failed", err);
  }
}

function updateDepthDownloadBtn() {
  const size = depthModelSelect.value;
  const downloaded = depthModelStatus[size];
  if (downloaded === false) {
    btnDownloadDepth.hidden = false;
    btnDownloadDepth.disabled = false;
    btnDownloadDepth.textContent = "Download";
  } else {
    btnDownloadDepth.hidden = true;
  }
}

btnDownloadDepth.addEventListener("click", async () => {
  const size = depthModelSelect.value;
  btnDownloadDepth.disabled = true;
  btnDownloadDepth.textContent = "Downloading...";
  try {
    const res = await fetch("/api/depth-models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: size }),
    });
    const data = await res.json();
    if (data.status === "ready") {
      depthModelStatus[size] = true;
      btnDownloadDepth.hidden = true;
    } else {
      schedulePoll(1500);
      const waitForDownload = async () => {
        const r = await fetch("/api/depth-models");
        if (!r.ok) return;
        const models = await r.json();
        for (const m of models) depthModelStatus[m.size] = m.downloaded;
        if (depthModelStatus[size]) {
          updateDepthDownloadBtn();
        } else {
          setTimeout(waitForDownload, 2000);
        }
      };
      setTimeout(waitForDownload, 2000);
    }
  } catch {
    btnDownloadDepth.disabled = false;
    btnDownloadDepth.textContent = "Download";
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
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = URL.createObjectURL(file);
  previewImg.src = previewObjectUrl;
  previewImg.classList.remove("hidden");
  dropZone.classList.add("has-preview");
  updateGenerateBtn();
}

// -- Settings ---------------------------------------------------------------

let saveSettingsTimer = null;

function numberInputValue(input, fallback) {
  const v = parseInt(input.value, 10);
  return Number.isFinite(v) && v >= 0 ? v : fallback;
}

function saveSettings() {
  if (saveSettingsTimer) clearTimeout(saveSettingsTimer);
  saveSettingsTimer = setTimeout(async () => {
    const body = {
      depth_model: depthModelSelect.value,
      max_dim: parseInt(maxDimSlider.value, 10),
      method: methodSelect.value,
      disable_cache: disableCacheCheck.checked,
    };
    // Only send cache limits once they have been populated from the server,
    // so an early save cannot overwrite them with blanks.
    if (sharpCacheMaxInput.value !== "") {
      body.sharp_cache_max_mb = numberInputValue(sharpCacheMaxInput, 0);
    }
    if (outputsKeepInput.value !== "") {
      body.outputs_keep = numberInputValue(outputsKeepInput, 0);
    }
    if (sharpIdleInput.value !== "") {
      body.sharp_idle_s = numberInputValue(sharpIdleInput, 0);
    }
    try {
      await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      // Limits apply immediately (pruning) - reflect the new sizes.
      loadCacheStats();
    } catch {
      // Best-effort persistence
    }
  }, 400);
}

async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    if (!res.ok) return;
    const s = await res.json();
    if (s.depth_model) depthModelSelect.value = s.depth_model;
    if (s.max_dim != null) {
      maxDimSlider.value = s.max_dim;
      maxDimValue.textContent = s.max_dim;
    }
    if (s.method && methodMeta[s.method]) methodSelect.value = s.method;
    else if (methodSelect.options.length) methodSelect.value = defaultMethodName;
    disableCacheCheck.checked = !!s.disable_cache;
    updateMethodInfo();
    updateDepthDownloadBtn();
  } catch {
    // Use defaults
  }
}

// -- Cache & memory ---------------------------------------------------------

function fillNumberInput(input, value) {
  // Do not clobber a value the user is editing right now.
  if (document.activeElement === input) return;
  if (value == null) return;
  input.value = String(Math.round(value));
}

function renderCacheStats(data) {
  if (!data) return;
  lastCacheStats = data;
  const sharp = data.sharp || {};
  const outputs = data.outputs || {};
  cacheSharpValue.textContent = sharp.error
    ? "unavailable"
    : fmtBytes(sharp.bytes || 0) + " - " + (sharp.files || 0) + " file" + (sharp.files === 1 ? "" : "s");
  cacheOutputsValue.textContent =
    fmtBytes(outputs.bytes || 0) + " - " + (outputs.files || 0) + " run" + (outputs.files === 1 ? "" : "s");
  const loaded = Array.isArray(data.loaded_models) ? data.loaded_models : [];
  loadedModelsValue.textContent = loaded.length ? loaded.join(", ") : "nothing loaded";
  btnClearSharp.disabled = !(sharp.files > 0);
  btnClearOutputs.disabled = !(outputs.files > 0);
  btnUnloadModels.disabled = loaded.length === 0;
  fillNumberInput(sharpCacheMaxInput, sharp.max_mb);
  fillNumberInput(outputsKeepInput, data.outputs_keep);
  fillNumberInput(sharpIdleInput, data.sharp_idle_s);
  cacheSharpPath.textContent = sharp.path || "";
  cacheOutputsPath.textContent = outputs.path || "";
}

function confirmClear(stats, what, unit, detail, consequence) {
  // Refuse to delete anything the server did not name a folder for.
  if (!stats || !stats.path) {
    setStatus("Cache folder unknown - nothing deleted.", false, true);
    return false;
  }
  const count = stats.files || 0;
  return confirm(
    "Delete " + count + " " + what + " " + unit + (count === 1 ? "" : "s") +
    " (" + fmtBytes(stats.bytes || 0) + ")?\n\n" +
    "Folder: " + stats.path + "\n" + detail + "\n\n" + consequence
  );
}

async function loadCacheStats() {
  try {
    const res = await fetch("/api/cache");
    if (!res.ok) return;
    renderCacheStats(await res.json());
  } catch {
    // Panel simply stays stale
  }
}

async function postCacheAction(url, body, button) {
  button.disabled = true;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      setStatus(msg, false, true);
      button.disabled = false;
      return null;
    }
    const data = await res.json();
    renderCacheStats(data);
    return data;
  } catch (err) {
    setStatus("Request failed: " + err.message, false, true);
    button.disabled = false;
    return null;
  }
}

btnClearSharp.addEventListener("click", async () => {
  const stats = lastCacheStats && lastCacheStats.sharp;
  if (!confirmClear(
    stats, "cached SHARP prediction", "file",
    "Only sharp_*.npz files directly in this folder are removed.",
    "The next run of each photo will predict again."
  )) return;
  const data = await postCacheAction("/api/cache/clear", { target: "sharp" }, btnClearSharp);
  if (data && data.deleted && data.deleted.sharp) {
    setStatus("SHARP cache cleared (" + fmtBytes(data.deleted.sharp.deleted_bytes) + ")", false, false);
  }
});

btnClearOutputs.addEventListener("click", async () => {
  const stats = lastCacheStats && lastCacheStats.outputs;
  if (!confirmClear(
    stats, "generated result", "folder",
    "Only the result folders inside it are removed.",
    "Images shown in the Output panel will stop loading."
  )) return;
  const data = await postCacheAction("/api/cache/clear", { target: "outputs" }, btnClearOutputs);
  if (data && data.deleted && data.deleted.outputs) {
    setStatus("Results cleared (" + fmtBytes(data.deleted.outputs.deleted_bytes) + ")", false, false);
  }
});

btnUnloadModels.addEventListener("click", async () => {
  const data = await postCacheAction("/api/models/unload", {}, btnUnloadModels);
  if (data) {
    const released = Array.isArray(data.released) ? data.released : [];
    setStatus(released.length ? "Unloaded: " + released.join(", ") : "Nothing was loaded", false, false);
  }
});

for (const input of [sharpCacheMaxInput, outputsKeepInput, sharpIdleInput]) {
  input.addEventListener("change", saveSettings);
}

disableCacheCheck.addEventListener("change", saveSettings);

maxDimSlider.addEventListener("input", () => {
  maxDimValue.textContent = maxDimSlider.value;
  saveSettings();
});

depthModelSelect.addEventListener("change", () => {
  updateDepthDownloadBtn();
  saveSettings();
});

methodSelect.addEventListener("change", () => {
  updateMethodInfo();
  updateGenerateBtn();
  syncModelGate(modelReady ? "ready" : "idle");
  updateSharpBanner();
  saveSettings();
  if (selectedMethodNeedsSharp() && !sharpReady) {
    schedulePoll(500);
  }
});

function updateTaichiStatus() {
  const hasTaichiMethods = Object.values(methodMeta).some((m) => m.uses_taichi);
  if (!hasTaichiMethods) {
    taichiStatus.hidden = true;
    taichiStatus.textContent = "";
    return;
  }
  taichiStatus.textContent = taichiRenderAvailable
    ? "Taichi GPU render: available on this machine."
    : "Taichi GPU render: not available here - (taichi) methods use torch.";
  taichiStatus.hidden = false;
}

function updateMethodInfo() {
  const method = methodSelect.value || defaultMethodName;
  const meta = methodMeta[method];
  let text = (meta && meta.ui_info) || "";
  const renderNote = taichiRenderNote(meta);
  if (renderNote) {
    text = text ? `${text}\n\n${renderNote}` : renderNote;
  }
  if (!text) {
    methodInfo.hidden = true;
    methodInfo.textContent = "";
  } else {
    methodInfo.textContent = text;
    methodInfo.hidden = false;
  }
  depthModelRow.hidden = !selectedMethodNeedsDepth();
  maxDimRow.hidden = !selectedMethodNeedsDepth();
}

function selectedMethodNeedsDepth() {
  const method = methodSelect.value || defaultMethodName;
  if (!(method in methodMeta)) return true;
  return methodMeta[method].needs_depth;
}

function selectedMethodNeedsSharp() {
  const method = methodSelect.value || defaultMethodName;
  if (!(method in methodMeta)) return false;
  return !methodMeta[method].needs_depth;
}

function updateSharpBanner() {
  if (!selectedMethodNeedsSharp()) {
    sharpBanner.hidden = true;
    return;
  }
  sharpBanner.hidden = false;
  if (sharpReady) {
    sharpBannerText.innerHTML =
      '<a href="https://apple.github.io/ml-sharp/" target="_blank" rel="noopener">SHARP</a> checkpoint ready.';
    btnDownloadSharp.hidden = true;
    btnCancelSharp.hidden = true;
    btnRemoveSharp.hidden = false;
    btnRemoveSharp.disabled = false;
    return;
  }
  if (sharpDownloading) {
    const pct = sharpPercent || 0;
    sharpBannerText.innerHTML = sharpQueued
      ? '<a href="https://apple.github.io/ml-sharp/" target="_blank" rel="noopener">SHARP</a> checkpoint queued - starts when the AI stereo models finish.'
      : 'Downloading <a href="https://apple.github.io/ml-sharp/" target="_blank" rel="noopener">SHARP</a> checkpoint... ' + pct + "%";
    btnDownloadSharp.hidden = true;
    btnCancelSharp.hidden = false;
    btnCancelSharp.disabled = false;
    btnRemoveSharp.hidden = true;
    return;
  }
  sharpBannerText.innerHTML =
    '<a href="https://apple.github.io/ml-sharp/" target="_blank" rel="noopener">SHARP</a> checkpoint required (2.8 GB, research-only).';
  btnDownloadSharp.hidden = false;
  btnDownloadSharp.disabled = false;
  btnDownloadSharp.textContent = "Download SHARP";
  btnCancelSharp.hidden = true;
  btnRemoveSharp.hidden = true;
}

btnDownloadSharp.addEventListener("click", async () => {
  // Optimistic UI - do not leave the button idle waiting on POST / stale polls.
  sharpDownloading = true;
  sharpQueued = modelDownloading;
  sharpPercent = sharpPercent || 0;
  sharpDownloadGraceUntil = Date.now() + 8000;
  updateSharpBanner();
  updateGenerateBtn();
  schedulePoll(1000);
  try {
    const res = await fetch("/api/model/download-sharp", { method: "POST" });
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      sharpDownloadGraceUntil = 0;
      sharpDownloading = false;
      sharpQueued = false;
      updateSharpBanner();
      btnDownloadSharp.textContent = "Retry";
      btnDownloadSharp.disabled = false;
      setStatus(msg, false, true);
      return;
    }
    const data = await res.json();
    applyModelStatus(data);
    schedulePoll(1500);
  } catch {
    sharpDownloadGraceUntil = 0;
    sharpDownloading = false;
    sharpQueued = false;
    updateSharpBanner();
    btnDownloadSharp.textContent = "Retry";
    btnDownloadSharp.disabled = false;
  }
});

btnCancelSharp.addEventListener("click", async () => {
  btnCancelSharp.disabled = true;
  sharpDownloadGraceUntil = 0;
  try {
    const res = await fetch("/api/model/cancel-sharp", { method: "POST" });
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      btnCancelSharp.disabled = false;
      setStatus(msg, false, true);
      return;
    }
    const data = await res.json();
    sharpDownloading = false;
    sharpQueued = false;
    sharpPercent = 0;
    applyModelStatus(data);
    schedulePoll(2000);
  } catch {
    btnCancelSharp.disabled = false;
  }
});

btnRemoveSharp.addEventListener("click", async () => {
  if (!confirm("Delete the SHARP checkpoint (2.8 GB)? You can re-download later.")) return;
  btnRemoveSharp.disabled = true;
  try {
    const res = await fetch("/api/model/delete-sharp", { method: "POST" });
    if (!res.ok) {
      let msg = `Server error (${res.status})`;
      try { msg = (await res.json()).error || msg; } catch {}
      btnRemoveSharp.disabled = false;
      setStatus(msg, false, true);
      return;
    }
    const data = await res.json();
    sharpReady = false;
    sharpDownloading = false;
    sharpPercent = 0;
    applyModelStatus(data);
    schedulePoll(2000);
  } catch {
    btnRemoveSharp.disabled = false;
  }
});

async function loadMethods() {
  try {
    const res = await fetch("/api/stereo-methods");
    if (!res.ok) {
      const text = await res.text();
      setStatus("Failed to load stereo methods (" + res.status + "). Restart PyStereo Web.", false, true);
      console.error("stereo-methods failed", res.status, text);
      return;
    }
    const data = await res.json();
    const methods = Array.isArray(data) ? data : data.methods;
    if (typeof data.taichi_render_available === "boolean") {
      taichiRenderAvailable = data.taichi_render_available;
    }
    if (!Array.isArray(methods) || methods.length === 0) {
      setStatus("No stereo methods available in this build.", false, true);
      return;
    }
    methodSelect.replaceChildren();
    for (const m of methods) {
      methodMeta[m.name] = m;
      if (m.default) defaultMethodName = m.name;
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = methodOptionLabel(m);
      methodSelect.appendChild(opt);
    }
    methodSelect.value = defaultMethodName;
    refreshMethodOptionLabels();
    updateTaichiStatus();
    updateMethodInfo();
  } catch (err) {
    setStatus("Failed to load stereo methods: " + err.message, false, true);
    console.error("stereo-methods failed", err);
  }
}

// -- Generate ---------------------------------------------------------------

function updateGenerateBtn() {
  const needsModel = selectedMethodNeedsDepth();
  const needsSharp = selectedMethodNeedsSharp();
  btnGenerate.disabled = !selectedFile || (needsModel && !modelReady) || (needsSharp && !sharpReady) || generating;
}

function setStatus(text, busy, isError) {
  statusBar.textContent = text;
  statusBar.className = "status-bar" + (isError ? " error" : busy ? " working" : "");
  statusWrap.classList.toggle("is-busy", !!busy);
  statusWrap.setAttribute("aria-busy", busy ? "true" : "false");
}

btnGenerate.addEventListener("click", async () => {
  const needsModel = selectedMethodNeedsDepth();
  if (!selectedFile || (needsModel && !modelReady) || generating) return;

  const depthModel = depthModelSelect.value;
  if (needsModel && depthModelStatus[depthModel] === false) {
    setStatus("Depth model not downloaded. Click Download first.", false, true);
    return;
  }

  generating = true;
  updateGenerateBtn();

  // Reset output
  pipelinePlaceholder.classList.add("hidden");
  pipelineStages.classList.remove("hidden");
  stageOriginal.hidden = false;
  if (stageOriginalObjectUrl) URL.revokeObjectURL(stageOriginalObjectUrl);
  stageOriginalObjectUrl = URL.createObjectURL(selectedFile);
  stageOriginalImg.src = stageOriginalObjectUrl;
  stageSplat.hidden = true;
  stageDepth.hidden = true;
  stageWarp.hidden = true;
  stageSbs.hidden = true;
  stageTiming.textContent = "";
  stageSteps.innerHTML = "";
  stageSteps.hidden = true;

  setStatus(needsModel ? "Running depth estimation + stereo synthesis..." : "Running SHARP stereo synthesis...", true, false);

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("method", methodSelect.value || defaultMethodName);
  form.append("max_dim", maxDimSlider.value);
  form.append("depth_model", depthModel);
  form.append("disable_cache", disableCacheCheck.checked ? "1" : "0");

  try {
    const res = await fetch("/api/generate", { method: "POST", body: form });

    if (!res.ok) {
      let msg = `Generation failed (${res.status})`;
      try {
        const err = await res.json();
        if (err.error) msg = err.error;
      } catch {}
      setStatus(msg, false, true);
      generating = false;
      updateGenerateBtn();
      return;
    }

    const data = await res.json();

    // Show splat render (SHARP methods)
    if (data.splat_url) {
      stageSplatImg.src = data.splat_url;
      stageSplat.hidden = false;
    }

    // Show depth
    if (data.depth_url) {
      stageDepthImg.src = data.depth_url;
      stageDepth.hidden = false;
    }

    // Show warp preview (pre-inpaint)
    if (data.warp_url) {
      stageWarpImg.src = data.warp_url;
      stageWarp.hidden = false;
    }

    // Show SBS
    if (data.sbs_url) {
      stageSbsImg.src = data.sbs_url;
      stageSbs.hidden = false;
    }

    // Per-step timing breakdown
    if (Array.isArray(data.timings) && data.timings.length) {
      for (const step of data.timings) {
        const li = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = step.label;
        const secs = document.createElement("span");
        secs.textContent = step.seconds.toFixed(2) + "s";
        li.append(label, secs);
        stageSteps.appendChild(li);
      }
      stageSteps.hidden = false;
    }

    const elapsed = data.elapsed_seconds != null ? data.elapsed_seconds.toFixed(1) : "?";
    const dims = data.width && data.height ? data.width + "x" + data.height : "";
    const methodUsed = data.method || "default";
    const backendNote =
      data.render_backend === "taichi" ? " - Taichi render" :
      data.render_backend === "torch" ? " - torch render" : "";
    const cacheNote =
      data.sharp_cache === "hit" ? " - SHARP cache hit" :
      data.sharp_cache === "off" ? " - cache off" :
      data.sharp_cache === "miss" ? " - SHARP cache miss" : "";
    stageTiming.textContent =
      "Method: " + methodUsed + " - " + dims + " - " + elapsed + "s total" + backendNote + cacheNote;

    setStatus("Done (" + elapsed + "s" + (backendNote + cacheNote).replaceAll(" - ", ", ") + ")", false, false);
  } catch (err) {
    setStatus("Request failed: " + err.message, false, true);
  }

  generating = false;
  updateGenerateBtn();
  loadCacheStats();
});

// -- Lightbox --------------------------------------------------------------

/** Stage images currently visible in the Output panel, in display order. */
let lightboxItems = [];
let lightboxIndex = 0;

function visibleStageImages() {
  return Array.from($$(".stage-card")).filter((card) => !card.hidden).map((card) => card.querySelector(".stage-img"));
}

function showLightboxItem(index) {
  const img = lightboxItems[index];
  if (!img) return;
  lightboxIndex = index;
  lightboxImg.src = img.currentSrc || img.src;
  lightboxImg.alt = img.alt;
  const label = img.closest(".stage-card").querySelector(".stage-label");
  lightboxCaption.textContent = label ? label.textContent : "";
  const multiple = lightboxItems.length > 1;
  lightboxPrev.hidden = !multiple;
  lightboxNext.hidden = !multiple;
}

function openLightbox(img) {
  lightboxItems = visibleStageImages();
  const index = lightboxItems.indexOf(img);
  if (index < 0) return;
  showLightboxItem(index);
  lightbox.hidden = false;
  document.body.classList.add("lightbox-open");
  lightboxClose.focus();
}

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.removeAttribute("src");
  document.body.classList.remove("lightbox-open");
}

function stepLightbox(delta) {
  if (lightboxItems.length < 2) return;
  const next = (lightboxIndex + delta + lightboxItems.length) % lightboxItems.length;
  showLightboxItem(next);
}

$$(".stage-img").forEach((img) => {
  img.addEventListener("click", () => openLightbox(img));
});

// Clicking the backdrop closes; clicks on the image or buttons do not.
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});
lightboxClose.addEventListener("click", closeLightbox);
lightboxPrev.addEventListener("click", () => stepLightbox(-1));
lightboxNext.addEventListener("click", () => stepLightbox(1));

document.addEventListener("keydown", (e) => {
  if (lightbox.hidden) return;
  if (e.key === "Escape") closeLightbox();
  else if (e.key === "ArrowLeft") stepLightbox(-1);
  else if (e.key === "ArrowRight") stepLightbox(1);
});

// -- Init -------------------------------------------------------------------

async function init() {
  await loadMethods();
  await loadDepthModels();
  await loadSettings();
  await loadCacheStats();
  await pollModelStatus();
}

init();
