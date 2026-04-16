const API_BASE = window.BACKEND_URL || "";

// Warm up the backend (wake from scale-to-zero) when page loads
fetch(API_BASE + "/health").catch(() => {});

let currentJobId = null;
let pollInterval = null;
let timeline = null;
let previewLoading = false;

const STAGE_ORDER = [
  "generating_music",
  "building_timeline",
  "searching_clips",
  "downloading_clips",
  "rendering_video",
];

const STAGE_LABELS = {
  generating_music: "Generate music",
  building_timeline: "Beat detection",
  searching_clips: "Search clips",
  downloading_clips: "Download clips",
  rendering_video: "Render",
};

const STATUS_MESSAGES = {
  generating_music: "Composing song via ElevenLabs...",
  building_timeline: "Detecting beats + building intensity timeline...",
  searching_clips: "Searching clips with turbopuffer hybrid search...",
  downloading_clips: "Downloading matched video clips...",
  rendering_video: "Rendering with speed ramping...",
};

// --- State ---

function showState(state) {
  document.getElementById("state-input").classList.toggle("hidden", state !== "input");
  document.getElementById("state-processing").classList.toggle("hidden", state !== "processing");
  document.getElementById("state-done").classList.toggle("hidden", state !== "done");
}

// --- Generate ---

async function generate() {
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;

  try {
    const res = await fetch(API_BASE + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode: getMode() }),
    });
    const data = await res.json();
    currentJobId = data.job_id;
    showState("processing");
    renderStages({});
    pollInterval = setInterval(pollStatus, 2000);
  } catch (err) {
    alert("Failed: " + err.message);
  }
}

// --- Poll ---

async function pollStatus() {
  if (!currentJobId) return;

  try {
    const res = await fetch(API_BASE + "/job/" + currentJobId);
    if (!res.ok) return;
    const data = await res.json();

    renderStages(data.stages || {});

    const stages = data.stages || {};
    let activeStage = null;
    for (const key of STAGE_ORDER) {
      if (stages[key] === "in_progress") activeStage = key;
    }
    document.getElementById("status-text").textContent =
      activeStage ? STATUS_MESSAGES[activeStage] : "Working...";

    if (data.status === "done") {
      clearInterval(pollInterval);
      pollInterval = null;
      document.getElementById("status-text").textContent = "Complete.";
      showDone();
      // Refresh the history log (new entry should appear)
      if (typeof window.refreshHistory === "function") {
        setTimeout(window.refreshHistory, 1500);
      }
      return;
    }

    if (data.status === "failed") {
      clearInterval(pollInterval);
      pollInterval = null;
      document.getElementById("status-text").textContent =
        "Error: " + (data.error || "Unknown");
      return;
    }

    const clipsDone =
      stages.searching_clips === "done" ||
      stages.downloading_clips === "in_progress" ||
      stages.downloading_clips === "done";

    if (clipsDone && !timeline) {
      loadPreview();
    }
  } catch (_) {}
}

// --- Stages ---

function renderStages(stages) {
  const container = document.getElementById("stages");
  container.innerHTML = STAGE_ORDER.map((key, i) => {
    const status = stages[key] || "pending";
    return `<div class="stage ${status}">
      <span class="stage-num">${String(i + 1).padStart(2, '0')}</span>
      <span class="stage-name">${STAGE_LABELS[key]}</span>
      <span class="stage-status"></span>
    </div>`;
  }).join("");
}

// --- Preview ---

function onTimeUpdate() {
  if (!timeline || !timeline.clips) return;
  const audio = document.getElementById("preview-audio");
  const video = document.getElementById("preview-video");
  const overlay = document.getElementById("lyric-overlay");
  const t = audio.currentTime * 1000;

  for (let i = timeline.clips.length - 1; i >= 0; i--) {
    const clip = timeline.clips[i];
    if (t >= clip.start_ms) {
      const clipSrc = clip.video_url || clip.url || "";
      if (clipSrc && video.getAttribute("data-current") !== clipSrc) {
        const fullSrc = clipSrc.startsWith("http") ? clipSrc : API_BASE + clipSrc;
        video.src = fullSrc;
        video.setAttribute("data-current", clipSrc);
        video.play().catch(() => {});
      }
      break;
    }
  }

  let lyric = "";
  for (let i = timeline.clips.length - 1; i >= 0; i--) {
    if (t >= timeline.clips[i].start_ms) {
      lyric = timeline.clips[i].lyric || "";
      break;
    }
  }
  overlay.textContent = lyric;
  overlay.style.opacity = lyric ? "1" : "0";
}

async function loadPreview() {
  if (previewLoading) return;
  previewLoading = true;
  try {
    const res = await fetch(API_BASE + "/preview/" + currentJobId);
    if (res.status === 202 || !res.ok) { previewLoading = false; return; }
    const data = await res.json();
    timeline = data;

    document.getElementById("preview-wrap").classList.remove("hidden");

    const audio = document.getElementById("preview-audio");
    if (data.audio_url) {
      audio.src = API_BASE + data.audio_url;
    }

    audio.removeEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("timeupdate", onTimeUpdate);
  } catch (_) {
    previewLoading = false;
  }
}

// --- Done ---

async function showDone() {
  showState("done");

  try {
    const res = await fetch(API_BASE + "/render/" + currentJobId);
    if (!res.ok) return;
    const data = await res.json();

    if (data.video_url) {
      // Rendi rendered — show video player
      document.getElementById("video-result").classList.remove("hidden");
      document.getElementById("done-video").src = data.video_url;
    } else {
      // Fallback — show download links + ffmpeg command
      document.getElementById("fallback-render").classList.remove("hidden");

      const grid = document.getElementById("asset-list");
      if (grid && data.assets) {
        let html = `<a class="asset-link" href="${API_BASE}${data.assets.audio}" download>
          <span class="label">Audio</span>audio.mp3</a>`;
        data.assets.clips.forEach((c, i) => {
          html += `<a class="asset-link" href="${API_BASE}${c.url}" download>
            <span class="label">Clip ${i + 1}</span>${c.filename}</a>`;
        });
        grid.innerHTML = html;
      }

      const cmdEl = document.getElementById("render-command");
      if (cmdEl && data.command) {
        cmdEl.textContent = data.command;
      }
    }
  } catch (_) {}
}

// --- Reset ---

function reset() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
  currentJobId = null;
  timeline = null;
  previewLoading = false;

  document.getElementById("preview-audio").removeEventListener("timeupdate", onTimeUpdate);
  document.getElementById("prompt").value = "";
  document.getElementById("stages").innerHTML = "";
  document.getElementById("status-text").textContent = "";
  document.getElementById("preview-wrap").classList.add("hidden");
  document.getElementById("preview-audio").src = "";
  document.getElementById("preview-video").src = "";
  document.getElementById("preview-video").removeAttribute("data-current");
  document.getElementById("lyric-overlay").textContent = "";
  document.getElementById("video-result").classList.add("hidden");
  document.getElementById("fallback-render").classList.add("hidden");
  document.getElementById("done-video").src = "";
  document.getElementById("asset-list").innerHTML = "";
  document.getElementById("render-command").textContent = "";

  showState("input");
}
