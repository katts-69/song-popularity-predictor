// Song Popularity Predictor: the page talks to the Flask API with fetch().
// Every request and response is visible in the browser's DevTools > Network tab.

const API = window.SONG_API_BASE;
const WAIT_LIMIT_MS = 90_000;   // the free Render server can take up to a minute to wake
const RETRY_GENRES_MS = 10_000;
const RING_LENGTH = 2 * Math.PI * 106; // circumference of the score ring in the SVG

// Sliders: what each one is, its range, and the example value it starts at.
const SLIDERS = [
  { id: "danceability", label: "Danceability", hint: "How easy it is to dance to", min: 0, max: 1, step: 0.01, value: 0.7 },
  { id: "energy", label: "Energy", hint: "How intense and busy it feels", min: 0, max: 1, step: 0.01, value: 0.65 },
  { id: "valence", label: "Mood", hint: "Sad (0) to happy (1)", min: 0, max: 1, step: 0.01, value: 0.55 },
  { id: "acousticness", label: "Acoustic", hint: "Electronic (0) to acoustic (1)", min: 0, max: 1, step: 0.01, value: 0.15 },
  { id: "instrumentalness", label: "Instrumental", hint: "Has vocals (0) to no vocals (1)", min: 0, max: 1, step: 0.01, value: 0 },
  { id: "speechiness", label: "Spoken words", hint: "Sung (0) to mostly talking (1)", min: 0, max: 1, step: 0.01, value: 0.05 },
  { id: "liveness", label: "Live sound", hint: "Studio (0) to live audience (1)", min: 0, max: 1, step: 0.01, value: 0.12 },
  { id: "loudness", label: "Loudness", hint: "Average volume in decibels", min: -40, max: 0, step: 0.5, value: -5.5, unit: " dB" },
  { id: "tempo", label: "Tempo", hint: "Beats per minute", min: 40, max: 220, step: 1, value: 118, unit: " BPM" },
  { id: "duration_ms", label: "Length", hint: "Minutes and seconds", min: 30_000, max: 600_000, step: 5_000, value: 200_000, format: "time" },
];
const KEYS = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"];
const LABELS = Object.fromEntries(SLIDERS.map((s) => [s.id, s.label]));
Object.assign(LABELS, { track_genre: "Genre", key: "Key", mode: "Mode", time_signature: "Beats per bar", explicit: "Explicit lyrics" });

const $ = (id) => document.getElementById(id);
const form = $("song-form");
const genreSelect = $("track_genre");
const predictButton = $("predict");
const serverStatus = $("server-status");
const formError = $("form-error");
const result = $("result");
const resultText = $("result-text");
const reasonsList = $("reasons");

// ---- Building the form --------------------------------------------------------

function formatTime(ms) {
  const total = Math.round(ms / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function formatSlider(slider, value) {
  if (slider.format === "time") return formatTime(value);
  const decimals = slider.step < 1 ? (slider.step < 0.1 ? 2 : 1) : 0;
  return Number(value).toFixed(decimals) + (slider.unit || "");
}

function buildSliders() {
  const box = $("sliders");
  for (const s of SLIDERS) {
    const wrap = document.createElement("div");
    wrap.className = "slider";
    wrap.innerHTML = `
      <div class="slider-head">
        <label for="${s.id}">${s.label}</label>
        <output id="${s.id}-out" for="${s.id}"></output>
      </div>
      <input type="range" id="${s.id}" name="${s.id}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}" aria-describedby="${s.id}-hint">
      <p class="hint" id="${s.id}-hint">${s.hint}</p>`;
    box.append(wrap);
    const input = wrap.querySelector("input");
    const output = wrap.querySelector("output");
    const show = () => {
      output.textContent = formatSlider(s, input.value);
      input.setAttribute("aria-valuetext", output.textContent);
    };
    input.addEventListener("input", show);
    show();
  }
  const keySelect = $("key");
  KEYS.forEach((name, i) => keySelect.add(new Option(name, i, false, i === 5)));
}

function readSong() {
  const song = { track_genre: genreSelect.value };
  for (const s of SLIDERS) song[s.id] = Number($(s.id).value);
  song.key = Number($("key").value);
  song.mode = Number($("mode").value);
  song.time_signature = Number($("time_signature").value);
  song.explicit = $("explicit").checked;
  return song;
}

// ---- Talking to the API ---------------------------------------------------------

async function callApi(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), WAIT_LIMIT_MS);
  try {
    const response = await fetch(API + path, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    return { ok: response.ok, status: response.status, body };
  } finally {
    clearTimeout(timer);
  }
}

// Shows "…12 s" while a slow request is running, so a sleeping server doesn't look broken.
function startElapsed(render) {
  const started = Date.now();
  render(0);
  const id = setInterval(() => render(Math.round((Date.now() - started) / 1000)), 1000);
  return () => clearInterval(id);
}

async function loadGenres() {
  const stop = startElapsed((s) => {
    serverStatus.textContent = s < 3
      ? "Connecting to the prediction server…"
      : `Waking the prediction server… ${s} s. The free server sleeps when nobody uses it and can take up to a minute.`;
  });
  try {
    const { ok, body } = await callApi("/genres");
    if (!ok) throw new Error("bad response");
    genreSelect.replaceChildren(...body.genres.map((g) => new Option(g, g, false, g === "pop")));
    genreSelect.disabled = false;
    predictButton.disabled = false;
    serverStatus.textContent = "Server ready.";
    loadRecent();
  } catch {
    serverStatus.textContent = `Can't reach the prediction server yet. Trying again in ${RETRY_GENRES_MS / 1000} seconds.`;
    $("recent-status").textContent = "Waiting for the prediction server.";
    setTimeout(loadGenres, RETRY_GENRES_MS);
  } finally {
    stop();
  }
}

// ---- Showing a prediction -------------------------------------------------------

function setDisc(score) {
  const ring = $("ring");
  const filled = score === null ? 0 : (score / 100) * RING_LENGTH;
  ring.style.strokeDasharray = `${filled} ${RING_LENGTH}`;
  $("score").textContent = score === null ? "–" : Math.round(score);
  $("disc-text").textContent = score === null ? "No prediction yet" : `Predicted popularity ${Math.round(score)} out of 100`;
}

function describeValue(feature, value) {
  if (feature === "duration_ms") return formatTime(value);
  if (feature === "key") return KEYS[value];
  if (feature === "mode") return value === 1 ? "Major" : "Minor";
  if (feature === "time_signature") return `${value}/4`;
  if (feature === "explicit") return value ? "Yes" : "No";
  if (feature === "track_genre") return value;
  const slider = SLIDERS.find((s) => s.id === feature);
  return slider ? formatSlider(slider, value) : String(value);
}

function reasonRow(name, detail, push, scale, extraClass = "") {
  const li = document.createElement("li");
  li.className = `reason ${extraClass}`.trim();
  const width = scale ? Math.min(50, (Math.abs(push) / scale) * 50) : 0;
  const direction = push >= 0 ? "up" : "down";
  const sign = push >= 0 ? "+" : "−";
  li.innerHTML = `
    <span class="reason-name"></span>
    <span class="bar-track" aria-hidden="true"><span class="bar ${direction}" style="width:${width}%"></span></span>
    <span class="reason-push">${sign}${Math.abs(push).toFixed(1)}</span>`;
  const nameEl = li.querySelector(".reason-name");
  nameEl.textContent = name;
  if (detail) {
    const small = document.createElement("small");
    small.textContent = detail;
    nameEl.append(small);
  }
  li.setAttribute("aria-label", `${name}${detail ? " " + detail : ""}: ${push >= 0 ? "pushes up" : "pulls down"} by ${Math.abs(push).toFixed(1)} points`);
  return li;
}

function showPrediction(data) {
  result.dataset.state = "ready";
  setDisc(data.prediction);
  const scale = Math.max(...data.reasons.map((r) => Math.abs(r.push)), Math.abs(data.other_features));
  resultText.innerHTML = "";
  const lead = document.createElement("p");
  lead.textContent = `Predicted popularity: ${data.prediction.toFixed(1)} out of 100.`;
  const how = document.createElement("p");
  how.className = "muted";
  how.textContent = `Every song starts at ${data.starting_point.toFixed(1)}, the average. Solid bars push the score up, outlined bars pull it down.`;
  resultText.append(lead, how);

  reasonsList.replaceChildren(
    ...data.reasons.map((r) => reasonRow(LABELS[r.feature] || r.feature, describeValue(r.feature, r.value), r.push, scale)),
    reasonRow("Everything else", `${15 - data.reasons.length} smaller features`, data.other_features, scale, "total"),
  );
  reasonsList.hidden = false;
}

function showResultError(message) {
  result.dataset.state = "error";
  setDisc(null);
  reasonsList.hidden = true;
  resultText.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = message;
  resultText.append(p);
}

// While predicting, the button stays focusable (disabling it would throw a keyboard
// user's focus away) and further presses are ignored.
let busy = false;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || predictButton.disabled) return;
  busy = true;
  predictButton.setAttribute("aria-disabled", "true");
  formError.hidden = true;
  result.dataset.state = "loading";
  reasonsList.hidden = true;
  const stop = startElapsed((s) => {
    resultText.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = s < 3 ? "Predicting…" : `Still working… ${s} s. If the server was asleep, the first prediction can take up to a minute.`;
    resultText.append(p);
  });

  try {
    const { ok, status, body } = await callApi("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readSong()),
    });
    stop();
    if (ok) {
      showPrediction(body);
      if (body.saved) loadRecent();
    } else if (status === 400) {
      formError.textContent = body.error || "Something in the form isn't valid.";
      formError.hidden = false;
      showResultError("Fix the problem above, then press Predict popularity again.");
    } else {
      showResultError(`The server answered with an error (${status}). Try again in a moment.`);
    }
  } catch {
    stop();
    showResultError("Couldn't reach the prediction server. It may still be waking up: wait a few seconds and press Predict popularity again.");
  } finally {
    busy = false;
    predictButton.removeAttribute("aria-disabled");
    if (window.matchMedia("(max-width: 860px)").matches) {
      const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      result.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
    }
  }
});

// ---- Recent predictions -----------------------------------------------------------

// The actual time in the visitor's own time zone. ("5 minutes ago" would depend on the
// visitor's clock being right, and a wrong clock made predictions look like the future.)
const WHEN = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const formatWhen = (iso) => WHEN.format(new Date(iso));

async function loadRecent() {
  const status = $("recent-status");
  const list = $("recent-list");
  try {
    const { ok, body } = await callApi("/recent?limit=8");
    if (!ok) {
      status.textContent = body.error || "Recent predictions are unavailable right now.";
      list.hidden = true;
      return;
    }
    if (body.predictions.length === 0) {
      status.textContent = "No predictions yet. Make one above and it will appear here.";
      list.hidden = true;
      return;
    }
    status.textContent = "";
    list.replaceChildren(...body.predictions.map((p) => {
      const li = document.createElement("li");
      const genre = document.createElement("span");
      genre.textContent = p.track_genre;
      const score = document.createElement("span");
      score.className = "recent-score";
      score.textContent = Math.round(p.prediction);
      const when = document.createElement("time");
      when.className = "recent-when";
      when.dateTime = p.created_at;
      when.textContent = formatWhen(p.created_at);
      li.append(genre, score, when);
      return li;
    }));
    list.hidden = false;
  } catch {
    status.textContent = "Recent predictions are unavailable right now.";
    list.hidden = true;
  }
}

buildSliders();
setDisc(null);
loadGenres();
