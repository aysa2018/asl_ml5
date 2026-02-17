// sketch.js (Aesthetic + video-sized canvas + overlay prediction)
// ml5 v1 HandPose + kNN (distance-weighted) classifier for ASL letters
// + localStorage persistence + undo + export/import JSON
// + NONE class + XY-only + rotation normalization + confidence+margin gating
// + record mode via "-" key, hold-to-add debounce, robust import
// + UPDATED: remove bottom info panel, crop canvas to video size, overlay prediction badge

let video;
let handPose;
let hands = [];

const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const NONE_LABEL = "NONE";
const ALL_LABELS = [...LABELS, NONE_LABEL];

// Data store
const examples = {};
ALL_LABELS.forEach((l) => (examples[l] = []));

// bumped because visuals + single-hand selection behavior is assumed
const STORAGE_KEY = "asl_handpose_examples_v5_cleanui_singlehand";

// Undo stack
let addHistory = [];

// File input for importing JSON backups
let importInput;

let isPredicting = true;

// Prediction smoothing state
let lastLabel = null;
let lastConf = 0;

let smoothQueue = [];
const SMOOTH_N = 12;
const STABLE_MIN = 9;

// kNN + gating settings
const K = 7;
const MIN_CONF = 0.62;
const MIN_MARGIN = 0.12;
const EPS = 1e-6;

let statusMsg = "loading HandPose...";

// Record mode ("-" key)
let recordMode = false;
let recordLabel = null;
let lastRecordAt = 0;
const RECORD_EVERY_MS = 140;
const ADD_DEBOUNCE_MS = 180;
let lastAddAt = 0;

const held = {};

// ----- SINGLE-HAND SELECTION -----
// If both hands are visible, we deterministically choose one.
const TRACK_HAND = "RIGHT"; // "RIGHT" or "LEFT"

// ----- Visual config -----
const VID_W = 640;
const VID_H = 480;

// Badge styling
const BADGE_SIZE = 92;
const BADGE_PAD = 14;

// Minimal HUD
const HUD_PAD = 12;
const HUD_H = 34;

function preload() {
  // maxHands: 2 so we can choose left/right deterministically, but we only USE ONE
  const options = { maxHands: 2, flipped: true };
  handPose = ml5.handPose(options);
}

function setup() {
  createCanvas(VID_W, VID_H);
  pixelDensity(1);

  importInput = createFileInput(handleImport, false);
  importInput.hide();
  importInput.elt.accept = ".json,application/json";

  const loaded = loadDataset();
  statusMsg = loaded
    ? `Loaded ✅ (${totalExamples()} ex)`
    : "Train NONE with N, then letters A–Z";

  video = createCapture(VIDEO);
  video.size(VID_W, VID_H);
  video.hide();

  handPose.detectStart(video, gotHands);

  textFont("system-ui, -apple-system, Segoe UI, Roboto, Arial");
}

function gotHands(results) {
  hands = results || [];
}

function draw() {
  background(0);

  // Webcam (mirrored)
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  pop();

  // subtle overlay for aesthetic contrast
  drawVignette();

  // hand points (clean + minimal)
  drawHandKeypoints();

  const feats = getHandFeaturesSingle();

  // record mode capture loop
  if (recordMode && recordLabel) {
    const now = millis();
    if (now - lastRecordAt >= RECORD_EVERY_MS) {
      const ok = addExample(recordLabel, { silent: true });
      lastRecordAt = now;

      if (ok) {
        statusMsg = `REC ● ${recordLabel === NONE_LABEL ? "NONE" : recordLabel} (+1)`;
      } else {
        statusMsg = "REC ● No hand detected";
      }
    }
  }

  // prediction
  if (isPredicting && feats && totalExamples() > 0) {
    const res = classifyKNN(feats);
    if (res) {
      lastLabel = res.label;
      lastConf = res.conf;

      smoothQueue.push(lastLabel);
      if (smoothQueue.length > SMOOTH_N) smoothQueue.shift();
    }
  }

  // overlay visuals
  drawPredictionBadge();
  drawHUD(feats);
}

/* -------------------- visuals -------------------- */

function drawVignette() {
  // soft dark corners
  push();
  noStroke();
  // top
  fill(0, 120);
  rect(0, 0, width, 40);
  // bottom
  fill(0, 90);
  rect(0, height - 50, width, 50);
  // left
  fill(0, 60);
  rect(0, 0, 24, height);
  // right
  fill(0, 60);
  rect(width - 24, 0, 24, height);
  pop();
}

function drawHUD(feats) {
  // minimal top-left pill
  const txt =
    (recordMode ? "REC ON" : "REC OFF") +
    `  |  predict ${isPredicting ? "ON" : "OFF"}` +
    `  |  hand ${feats ? "yes" : "no"}` +
    `  |  ex ${totalExamples()}`;

  push();
  noStroke();
  fill(0, 160);
  rect(HUD_PAD, HUD_PAD, textWidthSafe(txt) + 18, HUD_H, 999);

  fill(255);
  textAlign(LEFT, CENTER);
  textSize(13);
  text(txt, HUD_PAD + 10, HUD_PAD + HUD_H / 2 + 1);

  // transient status line (below pill)
  if (statusMsg) {
    fill(255, 200);
    textSize(12);
    text(statusMsg, HUD_PAD + 10, HUD_PAD + HUD_H + 14);
  }
  pop();
}

function drawPredictionBadge() {
  if (!isPredicting || totalExamples() === 0) return;

  const sm = getSmoothedLabel(); // { label, conf, stable }
  if (!sm.label) return;

  // hide NONE
  if (sm.label === NONE_LABEL) return;

  const x = width - BADGE_PAD - BADGE_SIZE;
  const y = BADGE_PAD;

  push();
  // drop shadow
  noStroke();
  fill(0, 140);
  rect(x + 3, y + 4, BADGE_SIZE, BADGE_SIZE, 18);

  // card
  fill(20, 20, 24, 210);
  rect(x, y, BADGE_SIZE, BADGE_SIZE, 18);

  // accent top strip
  fill(255, 255, 255, 18);
  rect(x, y, BADGE_SIZE, 18, 18, 18, 0, 0);

  // letter
  fill(255);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(52);
  text(sm.label, x + BADGE_SIZE / 2, y + BADGE_SIZE / 2 + 2);

  // confidence + stable
  textStyle(NORMAL);
  textSize(12);
  fill(220);
  const c = nf(sm.conf, 1, 2);
  const tag = sm.stable ? "stable" : "…";
  text(`${c} ${tag}`, x + BADGE_SIZE / 2, y + BADGE_SIZE - 16);

  pop();
}

function textWidthSafe(s) {
  push();
  textSize(13);
  const w = textWidth(s);
  pop();
  return w;
}

/* -------------------- hand selection + keypoints -------------------- */

function getWristX(hand) {
  const w = hand?.keypoints?.[0];
  return w?.x ?? 0;
}

function getValidHands(rawHands) {
  return (rawHands || []).filter((h) => h?.keypoints && h.keypoints.length === 21);
}

function sortHandsLeftToRight(rawHands) {
  const valid = getValidHands(rawHands);
  valid.sort((a, b) => getWristX(a) - getWristX(b));
  return valid;
}

function pickTrackedHand(rawHands) {
  const sorted = sortHandsLeftToRight(rawHands);
  if (sorted.length === 0) return null;
  if (sorted.length === 1) return sorted[0];
  return TRACK_HAND === "RIGHT" ? sorted[sorted.length - 1] : sorted[0];
}

function drawHandKeypoints() {
  const hand = pickTrackedHand(hands);
  if (!hand) return;

  push();
  // small, clean points
  stroke(255, 190);
  strokeWeight(5);

  for (const kp of hand.keypoints) {
    // because we displayed mirrored, mirror x for overlay
    const mx = width - kp.x;
    point(mx, kp.y);
  }
  pop();
}

function getLandmarks21(hand) {
  return hand.keypoints.map((k) => [k.x, k.y, k.z ?? 0]);
}

/* -------------------- features -------------------- */

function handToFeatsXYRotNorm(hand) {
  const lm = getLandmarks21(hand);

  const wrist = lm[0];
  const midMcp = lm[9];

  const wx = wrist[0], wy = wrist[1];
  const dx = midMcp[0] - wx;
  const dy = midMcp[1] - wy;

  const scale = Math.sqrt(dx * dx + dy * dy) || 1;

  const ang = Math.atan2(dy, dx);
  const rot = (-Math.PI / 2) - ang;
  const cosR = Math.cos(rot);
  const sinR = Math.sin(rot);

  const feats = [];
  for (let i = 0; i < 21; i++) {
    const x = (lm[i][0] - wx) / scale;
    const y = (lm[i][1] - wy) / scale;

    const xr = x * cosR - y * sinR;
    const yr = x * sinR + y * cosR;

    feats.push(xr, yr);
  }
  return feats;
}

function getHandFeaturesSingle() {
  const hand = pickTrackedHand(hands);
  if (!hand) return null;
  return handToFeatsXYRotNorm(hand);
}

/* -------------------- input -------------------- */

function keyTyped() {
  const k = key.toUpperCase();

  if (LABELS.includes(k)) {
    if (!recordMode) addExample(k);
    return;
  }

  if (k === "N") {
    if (!recordMode) addExample(NONE_LABEL);
    return;
  }

  if (k === ".") {
    undoLast();
    return;
  }

  if (k === " ") {
    isPredicting = !isPredicting;
    statusMsg = isPredicting ? "Prediction ON" : "Prediction OFF";
    return;
  }

  if (k === "1") {
    exportDataset();
    statusMsg = "Exported JSON ✅";
    return;
  }

  if (k === "0") {
    importInput.elt.value = "";
    importInput.show();
    statusMsg = "Choose a dataset JSON file…";
    return;
  }
}

function keyPressed() {
  if (keyCode === BACKSPACE) {
    clearAll();
    return false;
  }

  if (key === "-" || key === "_") {
    recordMode = !recordMode;
    if (!recordMode) {
      recordLabel = null;
      statusMsg = "Record mode OFF";
    } else {
      statusMsg = "Record mode ON — hold A–Z or N";
    }
    return false;
  }

  const k = key.toUpperCase();
  held[k] = true;

  if (recordMode) {
    if (LABELS.includes(k)) {
      recordLabel = k;
      lastRecordAt = 0;
      statusMsg = `REC ● ${k}`;
    } else if (k === "N") {
      recordLabel = NONE_LABEL;
      lastRecordAt = 0;
      statusMsg = "REC ● NONE";
    }
  }
}

function keyReleased() {
  const k = key.toUpperCase();
  held[k] = false;

  if (recordMode) {
    const wasLetter = LABELS.includes(k) && recordLabel === k;
    const wasNone = k === "N" && recordLabel === NONE_LABEL;
    if (wasLetter || wasNone) {
      recordLabel = null;
      statusMsg = "REC ● paused";
    }
  }
}

/* -------------------- dataset ops -------------------- */

function addExample(label, opts = {}) {
  const now = millis();
  if (!opts.silent && now - lastAddAt < ADD_DEBOUNCE_MS) return false;
  lastAddAt = now;

  const feats = getHandFeaturesSingle();
  if (!feats) {
    if (!opts.silent) statusMsg = "No hand detected";
    return false;
  }

  examples[label].push(feats);
  addHistory.push(label);
  saveDataset();

  if (!opts.silent) {
    statusMsg =
      label === NONE_LABEL
        ? `Added NONE (${examples[NONE_LABEL].length})`
        : `Added ${label} (${examples[label].length})`;
  }
  return true;
}

function undoLast() {
  if (addHistory.length === 0) {
    statusMsg = "Nothing to undo";
    return;
  }

  const label = addHistory.pop();
  if (examples[label].length > 0) {
    examples[label].pop();
    saveDataset();
    statusMsg = `Undo: ${label}`;
  } else {
    saveDataset();
    statusMsg = "Undo did nothing";
  }
}

function clearAll() {
  ALL_LABELS.forEach((l) => (examples[l] = []));
  addHistory = [];

  smoothQueue = [];
  lastLabel = null;
  lastConf = 0;

  recordLabel = null;
  recordMode = false;

  localStorage.removeItem(STORAGE_KEY);
  statusMsg = "Cleared dataset ✅";
}

function totalExamples() {
  return ALL_LABELS.reduce((sum, l) => sum + (examples[l]?.length || 0), 0);
}

/* -------------------- persistence -------------------- */

function saveDataset() {
  const payload = {
    version: 5,
    savedAt: new Date().toISOString(),
    examples: examples,
    addHistory: addHistory,
    meta: {
      feature: "xy_rot_norm_singlehand",
      trackHand: TRACK_HAND,
      dims: 42,
      k: K
    }
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function loadDataset() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return false;

  try {
    const payload = JSON.parse(raw);
    if (!payload.examples) return false;

    ALL_LABELS.forEach((l) => {
      const arr = payload.examples[l];
      examples[l] = Array.isArray(arr) ? arr : [];
    });

    addHistory = Array.isArray(payload.addHistory) ? payload.addHistory : [];

    smoothQueue = [];
    lastLabel = null;
    lastConf = 0;

    recordLabel = null;
    recordMode = false;

    return true;
  } catch (e) {
    return false;
  }
}

/* -------------------- export / import -------------------- */

function exportDataset() {
  const payload = {
    version: 5,
    savedAt: new Date().toISOString(),
    examples: examples,
    addHistory: addHistory,
    meta: {
      feature: "xy_rot_norm_singlehand",
      trackHand: TRACK_HAND,
      dims: 42,
      k: K
    }
  };
  saveJSON(payload, "asl_handpose_dataset_singlehand.json");
}

function parseMaybeJSON(file) {
  if (file && typeof file.data === "object" && file.data !== null) return file.data;

  if (file && typeof file.data === "string") {
    const s = file.data.trim();

    if (s.startsWith("data:")) {
      const comma = s.indexOf(",");
      if (comma === -1) throw new Error("Malformed data URL");
      const meta = s.slice(0, comma);
      const b64 = s.slice(comma + 1);

      if (meta.includes(";base64")) {
        const txt = atob(b64);
        return JSON.parse(txt);
      } else {
        const txt = decodeURIComponent(b64);
        return JSON.parse(txt);
      }
    }

    return JSON.parse(s);
  }

  throw new Error("File data was empty or unreadable");
}

function handleImport(file) {
  importInput.hide();

  if (!file) {
    statusMsg = "Import cancelled";
    return;
  }

  try {
    const payload = parseMaybeJSON(file);
    if (!payload || !payload.examples) throw new Error("Missing 'examples'");

    ALL_LABELS.forEach((l) => {
      const arr = payload.examples[l];
      examples[l] = Array.isArray(arr) ? arr : [];
    });

    addHistory = Array.isArray(payload.addHistory) ? payload.addHistory : [];

    saveDataset();

    smoothQueue = [];
    lastLabel = null;
    lastConf = 0;

    recordLabel = null;
    recordMode = false;

    statusMsg = `Imported ✅ (${totalExamples()} ex)`;
  } catch (e) {
    statusMsg = `Import failed — ${e.message || "invalid JSON"}`;
  }
}

/* -------------------- kNN (distance-weighted) + gating -------------------- */

function l2Distance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function classifyKNN(feats) {
  const neighbors = [];
  for (const label of ALL_LABELS) {
    const arr = examples[label];
    if (!arr || arr.length === 0) continue;
    for (let i = 0; i < arr.length; i++) {
      const d = l2Distance(feats, arr[i]);
      neighbors.push({ label, d });
    }
  }

  if (neighbors.length === 0) return null;

  neighbors.sort((a, b) => a.d - b.d);
  const k = Math.min(K, neighbors.length);
  const topK = neighbors.slice(0, k);

  const scores = {};
  for (const { label, d } of topK) {
    const w = 1 / (d + EPS);
    scores[label] = (scores[label] || 0) + w;
  }

  let bestLabel = null,
    bestScore = -Infinity;
  let secondLabel = null,
    secondScore = -Infinity;

  for (const label of Object.keys(scores)) {
    const s = scores[label];
    if (s > bestScore) {
      secondScore = bestScore;
      secondLabel = bestLabel;
      bestScore = s;
      bestLabel = label;
    } else if (s > secondScore) {
      secondScore = s;
      secondLabel = label;
    }
  }

  if (!isFinite(secondScore)) secondScore = 0;

  let total = 0;
  for (const s of Object.values(scores)) total += s;

  const conf = total > 0 ? bestScore / total : 0;
  const secondConf = total > 0 ? secondScore / total : 0;
  const margin = conf - secondConf;

  if (bestLabel === NONE_LABEL) {
    return { label: NONE_LABEL, conf, best: bestLabel, second: secondLabel, scores };
  }

  if (conf < MIN_CONF || margin < MIN_MARGIN) {
    return { label: NONE_LABEL, conf, best: bestLabel, second: secondLabel, scores };
  }

  return { label: bestLabel, conf, best: bestLabel, second: secondLabel, scores };
}

/* -------------------- smoothing -------------------- */

function getSmoothedLabel() {
  if (!lastLabel) return { label: null, conf: 0, stable: false };

  const counts = {};
  for (const l of smoothQueue) counts[l] = (counts[l] || 0) + 1;

  let best = null,
    bestCount = 0;
  for (const [l, c] of Object.entries(counts)) {
    if (c > bestCount) {
      best = l;
      bestCount = c;
    }
  }

  const stable = bestCount >= STABLE_MIN && lastConf >= MIN_CONF;
  return { label: best, conf: lastConf, stable };
}