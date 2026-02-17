// sketch.js
// ml5 v1 HandPose + kNN (distance-weighted) classifier for ASL letters
// + localStorage persistence + undo + export/import JSON
// + NONE class + better features (XY only + rotation normalization) + confidence+margin gating
// + NEW: record mode via "-" key (press/hold to capture), hold-to-add debounce, small UX fixes
//
// Keys:
// A–Z       → add example for that letter (auto-saves)
// N         → add example for NONE / REST class (auto-saves)
// "-"       → toggle RECORD mode (then HOLD a label key to record repeatedly) ✅
// "."       → undo last added example (auto-saves)
// SPACE     → toggle prediction
// BACKSPACE → clear dataset + delete saved data
// 1         → export dataset JSON (download)
// 0         → import dataset JSON (upload)
//
// Notes:
// - Start by training NONE a bunch (rest hand, transitions, off-pose).
// - Then train letters. Keep classes roughly balanced.
// - If it "gets worse" as you add data, it's usually because NONE is under-trained or data is messy.

let video;
let handPose;
let hands = [];

const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const NONE_LABEL = "NONE";
const ALL_LABELS = [...LABELS, NONE_LABEL];

// Data store
const examples = {};
ALL_LABELS.forEach((l) => (examples[l] = []));

const STORAGE_KEY = "asl_handpose_examples_v2";

// Undo stack: stores label for each added example, in order
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

// -------- NEW: Record mode ("-" key) --------
let recordMode = false;            // toggled by "-"
let recordLabel = null;            // currently held label in record mode
let lastRecordAt = 0;
const RECORD_EVERY_MS = 140;       // how fast we add examples while holding a label
const ADD_DEBOUNCE_MS = 180;       // prevents accidental double-adds on taps
let lastAddAt = 0;

// -------- NEW: simple key state --------
const held = {}; // held["A"]=true, held["N"]=true, etc.

function preload() {
  const options = { maxHands: 1, flipped: true };
  handPose = ml5.handPose(options);
}

function setup() {
  createCanvas(900, 650);

  // Hidden file picker for import
  importInput = createFileInput(handleImport);
  importInput.hide();

  // Load saved dataset (if any)
  const loaded = loadDataset();
  statusMsg = loaded
    ? `Loaded saved dataset ✅ (${totalExamples()} examples)`
    : "No saved dataset — train NONE with N, then letters A–Z";

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  handPose.detectStart(video, gotHands);
}

function gotHands(results) {
  hands = results || [];
}

function draw() {
  background(15);

  // Mirror video on canvas for a "selfie" view
  push();
  translate(650, 20);
  scale(-1, 1);
  image(video, 0, 0, 640, 480);
  pop();

  drawHandKeypoints();

  const feats = getHandFeatures();

  // -------- NEW: record mode capture loop --------
  if (recordMode && recordLabel) {
    const now = millis();
    if (now - lastRecordAt >= RECORD_EVERY_MS) {
      const ok = addExample(recordLabel, { silent: true }); // don't spam status
      lastRecordAt = now;

      if (ok) {
        statusMsg = `REC ● ${recordLabel === NONE_LABEL ? "NONE" : recordLabel} (+1)  |  total ${totalExamples()}`;
      } else {
        statusMsg = "REC ● No hand detected — adjust lighting / distance";
      }
    }
  }

  if (isPredicting && feats && totalExamples() > 0) {
    const res = classifyKNN(feats);
    if (res) {
      lastLabel = res.label;
      lastConf = res.conf;

      smoothQueue.push(lastLabel);
      if (smoothQueue.length > SMOOTH_N) smoothQueue.shift();
    }
  }

  drawUI(feats);
}

function drawHandKeypoints() {
  if (!hands.length) return;

  const hand = hands[0];
  if (!hand?.keypoints || hand.keypoints.length !== 21) return;

  push();
  translate(650, 20);
  stroke(255, 200);
  strokeWeight(6);

  for (const kp of hand.keypoints) {
    const mx = 640 - kp.x; // mirror x to match displayed video
    point(mx, kp.y);
  }

  pop();
}

function getLandmarks21(hand) {
  return hand.keypoints.map((k) => [k.x, k.y, k.z ?? 0]);
}

// --- Feature engineering: XY only + rotation normalization ---
// 1) translate by wrist
// 2) scale by wrist->middle_mcp distance
// 3) rotate so wrist->middle_mcp points "up" (negative Y)
function getHandFeatures() {
  if (!hands.length) return null;

  const hand = hands[0];
  if (!hand?.keypoints || hand.keypoints.length !== 21) return null;

  const lm = getLandmarks21(hand);

  const wrist = lm[0];
  const midMcp = lm[9];

  const wx = wrist[0], wy = wrist[1];
  const dx = midMcp[0] - wx;
  const dy = midMcp[1] - wy;

  const scale = Math.sqrt(dx * dx + dy * dy) || 1;

  // rotate so (dx,dy) becomes (0,-1)
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

// ---------- INPUT ----------
// NOTE: keyTyped doesn't fire reliably for "-" across layouts, so we handle record toggle in keyPressed.
function keyTyped() {
  const k = key.toUpperCase();

  // Train letters (tap-to-add, unless recordMode is ON)
  if (LABELS.includes(k)) {
    if (!recordMode) addExample(k);
    return;
  }

  // Train NONE (tap-to-add, unless recordMode is ON)
  if (k === "N") {
    if (!recordMode) addExample(NONE_LABEL);
    return;
  }

  // Undo last added
  if (k === ".") {
    undoLast();
    return;
  }

  // Toggle prediction
  if (k === " ") {
    isPredicting = !isPredicting;
    statusMsg = isPredicting ? "Prediction ON" : "Prediction OFF";
    return;
  }

  // Export dataset JSON
  if (k === "1") {
    exportDataset();
    statusMsg = "Exported dataset JSON ✅";
    return;
  }

  // Import dataset JSON
  if (k === "0") {
    importInput.elt.value = "";
    importInput.show();
    statusMsg = "Choose a dataset JSON file to import…";
    return;
  }
}

// Clear with BACKSPACE (and clear storage) + RECORD toggle + hold handling
function keyPressed() {
  // Prevent browser back-nav
  if (keyCode === BACKSPACE) {
    clearAll();
    return false;
  }

  // Toggle record mode with "-"
  // (key can be "-" or "_" depending on shift; also keyCode 189 in many browsers)
  if (key === "-" || key === "_") {
    recordMode = !recordMode;

    // when turning off, stop recording immediately
    if (!recordMode) {
      recordLabel = null;
      statusMsg = "Record mode OFF";
    } else {
      statusMsg = "Record mode ON — hold A–Z or N to record repeatedly";
    }
    return false;
  }

  // Track held keys for record mode
  const k = key.toUpperCase();
  held[k] = true;

  if (recordMode) {
    if (LABELS.includes(k)) {
      recordLabel = k;
      lastRecordAt = 0; // record immediately
      statusMsg = `REC ● ${k} (hold to record)`;
    } else if (k === "/") {
      recordLabel = NONE_LABEL;
      lastRecordAt = 0;
      statusMsg = `REC ● NONE (hold to record)`;
    }
  }
}

function keyReleased() {
  const k = key.toUpperCase();
  held[k] = false;

  // If the released key was driving recording, stop
  if (recordMode) {
    const wasLetter = LABELS.includes(k) && recordLabel === k;
    const wasNone = k === "N" && recordLabel === NONE_LABEL;
    if (wasLetter || wasNone) {
      recordLabel = null;
      statusMsg = "REC ● paused (hold a label key to continue)";
    }
  }
}

function addExample(label, opts = {}) {
  const now = millis();
  if (!opts.silent && now - lastAddAt < ADD_DEBOUNCE_MS) return false;
  lastAddAt = now;

  const feats = getHandFeatures();
  if (!feats) {
    if (!opts.silent) statusMsg = "No hand detected — move closer, better light, open palm first";
    return false;
  }

  examples[label].push(feats);
  addHistory.push(label);
  saveDataset();

  if (!opts.silent) {
    if (label === NONE_LABEL) {
      statusMsg = `Added example for NONE (now ${examples[NONE_LABEL].length}) — saved ✅`;
    } else {
      statusMsg = `Added example for ${label} (now ${examples[label].length}) — saved ✅`;
    }
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
    statusMsg = `Undid last add: ${label} — saved ✅`;
  } else {
    saveDataset();
    statusMsg = "Undo did nothing (dataset already empty)";
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
  statusMsg = "Cleared dataset + deleted saved data ✅";
}

function totalExamples() {
  return ALL_LABELS.reduce((sum, l) => sum + (examples[l]?.length || 0), 0);
}

// ---------- Persistence ----------
function saveDataset() {
  const payload = {
    version: 2,
    savedAt: new Date().toISOString(),
    examples: examples,
    addHistory: addHistory,
    meta: {
      feature: "xy_rot_norm",
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

// ---------- Export / Import ----------
function exportDataset() {
  const payload = {
    version: 2,
    savedAt: new Date().toISOString(),
    examples: examples,
    addHistory: addHistory,
    meta: {
      feature: "xy_rot_norm",
      k: K
    }
  };
  saveJSON(payload, "asl_handpose_dataset.json");
}

function handleImport(file) {
  importInput.hide();

  if (!file || !file.data) {
    statusMsg = "Import cancelled";
    return;
  }

  try {
    const payload = JSON.parse(file.data);
    if (!payload.examples) throw new Error("Missing examples");

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

    statusMsg = `Imported ✅ (${totalExamples()} examples)`;
  } catch (e) {
    statusMsg = "Import failed — invalid JSON";
  }
}

// -------- kNN classifier (distance-weighted) + gating --------
function l2Distance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

// Returns: { label, conf, best, second, scores }
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

  let bestLabel = null;
  let bestScore = -Infinity;
  let secondLabel = null;
  let secondScore = -Infinity;

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

  // If NONE wins, output NONE
  if (bestLabel === NONE_LABEL) {
    return { label: NONE_LABEL, conf, best: bestLabel, second: secondLabel, scores };
  }

  // If letter wins but low conf or low margin, output NONE
  if (conf < MIN_CONF || margin < MIN_MARGIN) {
    return { label: NONE_LABEL, conf, best: bestLabel, second: secondLabel, scores };
  }

  return { label: bestLabel, conf, best: bestLabel, second: secondLabel, scores };
}

// -------- UI --------
function getSmoothedLabel() {
  if (!lastLabel) return { label: null, conf: 0, stable: false };

  const counts = {};
  for (const l of smoothQueue) counts[l] = (counts[l] || 0) + 1;

  let best = null, bestCount = 0;
  for (const [l, c] of Object.entries(counts)) {
    if (c > bestCount) {
      best = l;
      bestCount = c;
    }
  }

  const stable = bestCount >= STABLE_MIN && lastConf >= MIN_CONF;
  return { label: best, conf: lastConf, stable };
}

function drawUI(feats) {
  fill(25);
  noStroke();
  rect(20, 520, 860, 120, 16);

  fill(240);
  textSize(16);
  textAlign(LEFT, TOP);

  text(
    `Status: ${statusMsg}\n` +
      `Train: A–Z | NONE: N | Record mode: "-" (${recordMode ? "ON" : "OFF"}) | Undo: . | Export: 1 | Import: 0 | [space] predict ${isPredicting ? "ON" : "OFF"} | Backspace: clear\n` +
      `Hand: ${feats ? "yes" : "no"} | Total: ${totalExamples()} | NONE: ${examples[NONE_LABEL].length}\n` +
      `kNN: k=${K} | gate: conf>=${MIN_CONF}, margin>=${MIN_MARGIN} | REC every ${RECORD_EVERY_MS}ms`,
    35,
    535
  );

  const sm = getSmoothedLabel();
  textSize(30);
  const shown = sm.label ? sm.label : "—";
  const shownPretty = shown === NONE_LABEL ? "NONE" : shown;

  text(`Prediction: ${shownPretty}${sm.stable ? " (stable)" : ""}`, 35, 595);

  textSize(16);
  text(`conf: ${nf(sm.conf, 1, 2)}`, 320, 605);

  fill(200);
  textSize(14);
  text(
    "Tip: Train NONE a lot (rest + transitions). If it misfires, add more NONE and/or raise MIN_CONF/MIN_MARGIN.",
    35,
    622
  );
}