// sketch.js — ASL Trainer (2 Modes + Home Screen) — Home button INSIDE camera
// Change in this version:
// ✅ “← Home” is drawn as an in-canvas button (bottom-left) on camera screens (AZ/WORD + Congrats)
// ✅ Removed DOM home button (so it’s truly inside the camera screen)
// Keeps: centered home mode picker, A–Z [i/26] + progress bar, bigger congrats flashcards

// ---------- Crash logger ----------
window.addEventListener("error", (e) => console.error("WINDOW ERROR:", e.error || e.message, e));
window.addEventListener("unhandledrejection", (e) => console.error("PROMISE REJECTION:", e.reason, e));

let video;
let handPose;
let hands = [];

const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const NONE_LABEL = "NONE";
const ALL_LABELS = [...LABELS, NONE_LABEL];

// Data store
const examples = {};
ALL_LABELS.forEach((l) => (examples[l] = []));

const STORAGE_KEY = "asl_handpose_v9_modes_singlehand_incanvas_home";

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

// Optional: require a bit more confidence in game progression
const GAME_MIN_CONF = 0.68;

let statusMsg = "loading HandPose...";

// Record mode ("-" key)
let recordMode = false;
let recordLabel = null;
let lastRecordAt = 0;
const RECORD_EVERY_MS = 140;
const ADD_DEBOUNCE_MS = 180;
let lastAddAt = 0;

const held = {};

// Single-hand selection
const TRACK_HAND = "RIGHT"; // "RIGHT" or "LEFT"

// Visual config
const VID_W = 640;
const VID_H = 480;

// Top-right UI
const PANEL_PAD = 14;
const PANEL_GAP = 10;
const REF_SIZE = 92;
const BADGE_SIZE = 92;

// Game logic
let alphaImgs = {}; // {A: p5.Image|null, ...}

const ADVANCE_HOLD_MS = 600;
let correctStartAt = null;

const ADVANCE_COOLDOWN_MS = 650;
let lastAdvanceAt = -1e9;

// ---------- Modes ----------
const MODE_HOME = "HOME";
const MODE_AZ = "AZ";
const MODE_WORD = "WORD";
let mode = MODE_HOME;

// A–Z mode state
let azActive = false;
let azIdx = 0; // 0..25
let azStreak = 0;
let azSolved = 0;

// Word mode state
let wordActive = false;
let wordText = "EAT";
let wordIdx = 0;
let wordStreak = 0;
let wordSolved = 0;
let wordCompleted = false;

// ---------- DOM UI ----------
let uiHomeWrap, btnModeAZ, btnModeWord;
let uiWordWrap, wordInput, btnStartWord;

// ---------- In-canvas Home button ----------
const HOME_BTN = { x: 12, y: VID_H - 12 - 40, w: 112, h: 40, label: "← Home" };

// --- preload ---
function preload() {
  const options = { maxHands: 2, flipped: true };
  handPose = ml5.handPose(options);

  for (const L of LABELS) {
    const path = `Alphabets/${L}.png`;
    alphaImgs[L] = loadImage(
      path,
      () => {},
      (err) => {
        console.warn("Failed to load alphabet image:", path, err);
        alphaImgs[L] = null;
      }
    );
  }
}

function setup() {
  createCanvas(VID_W, VID_H);
  pixelDensity(1);
  textFont("system-ui, -apple-system, Segoe UI, Roboto, Arial");

  // Import
  importInput = createFileInput(handleImport, false);
  importInput.hide();
  importInput.elt.accept = ".json,application/json";

  const loaded = loadDataset();
  statusMsg = loaded ? `Loaded ✅ (${totalExamples()} ex)` : "No saved dataset — train NONE (N), then letters (A–Z)";

  // Webcam
  video = createCapture(VIDEO);
  video.size(VID_W, VID_H);
  video.hide();
  handPose.detectStart(video, gotHands);

  setupUI();
  setMode(MODE_HOME);
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

  drawVignette();

  const feats = getHandFeaturesSingle();

  // record mode capture loop (training)
  if (recordMode && recordLabel) {
    const now = millis();
    if (now - lastRecordAt >= RECORD_EVERY_MS) {
      const ok = addExample(recordLabel, { silent: true });
      lastRecordAt = now;
      statusMsg = ok ? `REC ● ${recordLabel} (+1)` : "REC ● No hand detected";
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

  // mode logic
  if (mode === MODE_AZ && azActive) updateProgressForTarget(currentTargetAZ());
  if (mode === MODE_WORD && wordActive && !wordCompleted) updateProgressForTarget(currentTargetWord());

  // overlays
  if (mode === MODE_HOME) {
    drawHomeOverlay(feats);
  } else if (mode === MODE_AZ) {
    const done = !azActive && azIdx >= 25;
    drawRoundOverlay({
      feats,
      title: "A–Z MODE",
      step: azIdx + 1,
      total: 26,
      streak: azStreak,
      active: azActive,
      done,
      showProgressBar: true,
    });
    if (done) drawCompletionFlashcards("A–Z", LABELS.join(""));
  } else if (mode === MODE_WORD) {
    const total = wordText.length || 1;
    drawRoundOverlay({
      feats,
      title: "WORD MODE",
      step: wordIdx + 1,
      total,
      streak: wordStreak,
      active: wordActive,
      done: wordCompleted,
      showProgressBar: false,
    });
    if (wordCompleted) drawCompletionFlashcards(wordText, wordText);
  }

  drawHUDMinimal(feats);

  // ✅ In-canvas Home button on camera screens (including congrats overlay)
  if (mode !== MODE_HOME) drawHomeButton();
}

/* ===================== UI / MODES ===================== */

function setupUI() {
  // HOME wrap centered
  uiHomeWrap = createDiv();
  uiHomeWrap.style("position", "absolute");
  uiHomeWrap.style("left", "50%");
  uiHomeWrap.style("top", "50%");
  uiHomeWrap.style("transform", "translate(-50%, -50%)");
  uiHomeWrap.style("padding", "16px");
  uiHomeWrap.style("border-radius", "16px");
  uiHomeWrap.style("background", "rgba(0,0,0,0.62)");
  uiHomeWrap.style("color", "white");
  uiHomeWrap.style("font-family", "system-ui, -apple-system, Segoe UI, Roboto, Arial");
  uiHomeWrap.style("text-align", "center");

  const title = createDiv("Choose a mode");
  title.parent(uiHomeWrap);
  title.style("font-weight", "800");
  title.style("font-size", "18px");
  title.style("margin-bottom", "12px");

  const row = createDiv();
  row.parent(uiHomeWrap);
  row.style("display", "flex");
  row.style("gap", "10px");
  row.style("justify-content", "center");

  btnModeAZ = createButton("A–Z Mode");
  btnModeAZ.parent(row);
  styleButton(btnModeAZ);
  btnModeAZ.mousePressed(() => {
    startAZ();
    setMode(MODE_AZ);
  });

  btnModeWord = createButton("Word Mode");
  btnModeWord.parent(row);
  styleButton(btnModeWord);
  btnModeWord.mousePressed(() => setMode(MODE_WORD));

  // WORD wrap (top-left)
  uiWordWrap = createDiv();
  uiWordWrap.style("position", "absolute");
  uiWordWrap.style("left", "16px");
  uiWordWrap.style("top", "16px");
  uiWordWrap.style("padding", "12px");
  uiWordWrap.style("border-radius", "14px");
  uiWordWrap.style("background", "rgba(0,0,0,0.55)");
  uiWordWrap.style("color", "white");
  uiWordWrap.style("font-family", "system-ui, -apple-system, Segoe UI, Roboto, Arial");

  const wtitle = createDiv("Enter a word (A–Z only)");
  wtitle.parent(uiWordWrap);
  wtitle.style("font-weight", "700");
  wtitle.style("margin-bottom", "8px");

  wordInput = createInput("Eat");
  wordInput.parent(uiWordWrap);
  wordInput.style("padding", "8px 10px");
  wordInput.style("border-radius", "10px");
  wordInput.style("border", "1px solid rgba(255,255,255,0.25)");
  wordInput.style("background", "rgba(20,20,24,0.85)");
  wordInput.style("color", "white");
  wordInput.style("width", "160px");
  wordInput.input(() => {
    const cleaned = sanitizeWord(wordInput.value());
    statusMsg = cleaned ? `Word Mode ready: ${cleaned}` : "Word Mode: type letters A–Z";
  });

  btnStartWord = createButton("Start");
  btnStartWord.parent(uiWordWrap);
  styleButton(btnStartWord);
  btnStartWord.style("margin-left", "8px");
  btnStartWord.mousePressed(() => {
    const cleaned = sanitizeWord(wordInput.value());
    if (!cleaned) {
      statusMsg = "Please enter a word using A–Z letters";
      return;
    }
    startWord(cleaned);
    setMode(MODE_WORD);
  });
}

function styleButton(b) {
  b.style("padding", "8px 12px");
  b.style("border-radius", "12px");
  b.style("border", "1px solid rgba(255,255,255,0.18)");
  b.style("background", "rgba(20,20,24,0.85)");
  b.style("color", "white");
  b.style("cursor", "pointer");
}

function setMode(next) {
  mode = next;

  if (mode === MODE_HOME) {
    uiHomeWrap.show();
    uiWordWrap.hide();
    statusMsg = statusMsg || "Choose a mode";
  } else if (mode === MODE_AZ) {
    uiHomeWrap.hide();
    uiWordWrap.hide();
    statusMsg = "A–Z Mode — press ENTER to restart round";
  } else if (mode === MODE_WORD) {
    uiHomeWrap.hide();
    uiWordWrap.show();
    if (!wordActive) statusMsg = "Word Mode — enter a word, then Start";
  }

  correctStartAt = null;
  lastAdvanceAt = -1e9;
}

function sanitizeWord(s) {
  return String(s || "").toUpperCase().replace(/[^A-Z]/g, "");
}

/* ===================== In-canvas Home button ===================== */

function drawHomeButton() {
  const hovering = isPointInRect(mouseX, mouseY, HOME_BTN);
  push();
  noStroke();

  // shadow
  fill(0, 150);
  rect(HOME_BTN.x + 3, HOME_BTN.y + 4, HOME_BTN.w, HOME_BTN.h, 14);

  // button
  fill(hovering ? 30 : 20, hovering ? 30 : 20, hovering ? 34 : 24, 220);
  rect(HOME_BTN.x, HOME_BTN.y, HOME_BTN.w, HOME_BTN.h, 14);

  // subtle top highlight
  fill(255, 255, 255, 18);
  rect(HOME_BTN.x, HOME_BTN.y, HOME_BTN.w, 14, 14, 14, 0, 0);

  fill(255);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(13);
  text(HOME_BTN.label, HOME_BTN.x + HOME_BTN.w / 2, HOME_BTN.y + HOME_BTN.h / 2 + 1);
  pop();
}

function mousePressed() {
  if (mode !== MODE_HOME && isPointInRect(mouseX, mouseY, HOME_BTN)) {
    setMode(MODE_HOME);
    return false;
  }
}

function isPointInRect(px, py, r) {
  return px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h;
}

/* ===================== A–Z MODE ===================== */

function startAZ() {
  azActive = true;
  azIdx = 0;
  azStreak = 0;
  azSolved = 0;
  correctStartAt = null;
  lastAdvanceAt = -1e9;
  statusMsg = "A–Z started — sign A";
}

function currentTargetAZ() {
  return LABELS[constrain(azIdx, 0, 25)];
}

function advanceAZ(wasCorrect) {
  lastAdvanceAt = millis();
  correctStartAt = null;

  if (wasCorrect) {
    azStreak += 1;
    azSolved += 1;
    statusMsg = `✅ Correct! ${currentTargetAZ()}  (streak ${azStreak})`;
  } else {
    azStreak = 0;
    statusMsg = `Skipped → ${currentTargetAZ()}`;
  }

  azIdx += 1;

  if (azIdx >= LABELS.length) {
    azActive = false;
    azIdx = LABELS.length - 1;
    statusMsg = `🎉 Completed A–Z! Solved: ${azSolved}/26`;
  }
}

/* ===================== WORD MODE ===================== */

function startWord(cleaned) {
  wordText = cleaned;
  wordActive = true;
  wordCompleted = false;
  wordIdx = 0;
  wordStreak = 0;
  wordSolved = 0;
  correctStartAt = null;
  lastAdvanceAt = -1e9;
  statusMsg = `Word started — sign ${wordText[0]}`;
}

function currentTargetWord() {
  if (!wordText || wordText.length === 0) return "—";
  return wordText[constrain(wordIdx, 0, wordText.length - 1)];
}

function advanceWord(wasCorrect) {
  lastAdvanceAt = millis();
  correctStartAt = null;

  if (wasCorrect) {
    wordStreak += 1;
    wordSolved += 1;
    statusMsg = `✅ Correct! ${currentTargetWord()}  (streak ${wordStreak})`;
  } else {
    wordStreak = 0;
    statusMsg = `Skipped → ${currentTargetWord()}`;
  }

  wordIdx += 1;

  if (wordIdx >= wordText.length) {
    wordCompleted = true;
    wordActive = false;
    wordIdx = max(0, wordText.length - 1);
    statusMsg = `🎉 Nice! You signed: ${wordText}`;
  }
}

/* ===================== PROGRESS CHECK (shared) ===================== */

function updateProgressForTarget(targetLetter) {
  const sm = getSmoothedLabel();

  const ok =
    sm.stable &&
    sm.label === targetLetter &&
    sm.conf >= GAME_MIN_CONF &&
    millis() - lastAdvanceAt >= ADVANCE_COOLDOWN_MS;

  if (ok) {
    if (correctStartAt == null) correctStartAt = millis();

    if (millis() - correctStartAt >= ADVANCE_HOLD_MS) {
      if (mode === MODE_AZ) advanceAZ(true);
      if (mode === MODE_WORD) advanceWord(true);
    }
  } else {
    correctStartAt = null;
  }
}

/* ===================== OVERLAYS ===================== */

function drawHomeOverlay(feats) {
  push();
  noStroke();
  fill(0, 150);
  rect(12, height - 70, width - 24, 52, 16);

  fill(255);
  textAlign(LEFT, CENTER);
  textSize(13);
  text("Train first: add NONE (N) + letters (A–Z). Then pick a mode.\nTip: Hold '-' (record mode) to add examples quickly.", 26, height - 44);
  pop();
}

function drawRoundOverlay({ feats, title, step, total, streak, active, done, showProgressBar }) {
  // Top-right stacked layout: target image (top) + prediction badge (below)
  const xRight = width - PANEL_PAD;
  const yTop = PANEL_PAD;

  const xRef = xRight - REF_SIZE;
  const yRef = yTop;

  // Target image (NO frame)
  drawTargetImage(xRef, yRef, REF_SIZE, REF_SIZE, title);

  // Prediction below
  const xBadge = xRight - BADGE_SIZE;
  const yBadge = yRef + REF_SIZE + PANEL_GAP;
  drawPredictionBadgeAt(xBadge, yBadge, BADGE_SIZE, BADGE_SIZE);

  // A–Z progress bar + [i/26]
  if (showProgressBar) {
    const barW = REF_SIZE;
    const barH = 7;
    const barX = xRef;
    const barY = yBadge + BADGE_SIZE + 14;

    const t = total <= 1 ? 1 : constrain((step - 1) / (total - 1), 0, 1);
    drawProgressBar(barX, barY, barW, barH, t);

    push();
    fill(255, 220);
    textAlign(RIGHT, TOP);
    textSize(12);
    text(`[${step}/${total}]`, width - PANEL_PAD, barY + 10);
    pop();
  }

  // Progress line
  const progY = yBadge + BADGE_SIZE + (showProgressBar ? 34 : 10);
  push();
  fill(255, 210);
  textAlign(RIGHT, TOP);
  textSize(12);
  text(`${title}  |  ${step}/${total}  |  streak ${streak}`, width - PANEL_PAD, progY);
  pop();

  // Hold bar (tucked above home button area)
  if (active && correctStartAt != null) {
    const t = constrain((millis() - correctStartAt) / ADVANCE_HOLD_MS, 0, 1);
    drawHoldBar(xRef, height - 12 - 6, REF_SIZE, 6, t);
  }

  // Empty dataset hint
  if (totalExamples() === 0) {
    push();
    noStroke();
    fill(0, 170);
    rect(PANEL_PAD, height - 56, width - PANEL_PAD * 2, 42, 16);
    fill(255);
    textAlign(LEFT, CENTER);
    textSize(13);
    text("Train first: add NONE (N) + letters (A–Z), then play.", PANEL_PAD + 14, height - 35);
    pop();
  }

  // Done hint
  if (done) {
    push();
    noStroke();
    fill(0, 170);
    rect(12, 86, 300, 40, 14);
    fill(255);
    textAlign(LEFT, CENTER);
    textSize(13);
    text("Round finished — press ENTER to play again", 24, 106);
    pop();
  }
}

function drawCompletionFlashcards(title, lettersStr) {
  push();
  noStroke();
  fill(0, 185);
  rect(0, 0, width, height);

  fill(20, 20, 24, 240);
  rect(26, 40, width - 52, height - 80, 22);

  fill(255);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  textSize(24);
  text("🎉 Great job!", width / 2, 62);

  textStyle(NORMAL);
  textSize(14);
  fill(235);
  text(`You completed: ${title}`, width / 2, 96);

  const letters = lettersStr.split("").filter((c) => LABELS.includes(c));

  // big grid up to 12
  const maxShow = min(12, letters.length);
  const start = max(0, letters.length - maxShow);
  const shown = letters.slice(start);

  const cols = min(6, shown.length);
  const rows = Math.ceil(shown.length / cols);

  const cardW = 110;
  const cardH = 110;
  const gap = 12;

  const gridW = cols * cardW + (cols - 1) * gap;
  const gridH = rows * cardH + (rows - 1) * gap;

  let gx = width / 2 - gridW / 2;
  let gy = 132;

  for (let i = 0; i < shown.length; i++) {
    const L = shown[i];
    const cx = gx + (i % cols) * (cardW + gap);
    const cy = gy + Math.floor(i / cols) * (cardH + gap);

    fill(0, 150);
    rect(cx + 3, cy + 4, cardW, cardH, 14);
    fill(20, 20, 24, 230);
    rect(cx, cy, cardW, cardH, 14);

    const img = alphaImgs[L];
    if (img && img.width > 0) {
      const pad = 8;
      const s = min((cardW - pad * 2) / img.width, (cardH - pad * 2) / img.height);
      image(img, cx + (cardW - img.width * s) / 2, cy + (cardH - img.height * s) / 2, img.width * s, img.height * s);
    } else {
      fill(255);
      textAlign(CENTER, CENTER);
      textStyle(BOLD);
      textSize(22);
      text(L, cx + cardW / 2, cy + cardH / 2 + 1);
    }
  }

  // Big word/sequence at bottom
  fill(255);
  textAlign(CENTER, BOTTOM);
  textStyle(BOLD);
  textSize(30);
  text(lettersStr, width / 2, height - 64);

  textStyle(NORMAL);
  textSize(12);
  fill(230);
  text("Tap ← Home (bottom-left) for the next round.", width / 2, height - 40);

  pop();
}

function drawTargetImage(x, y, w, h, headerLabel) {
  const L = mode === MODE_WORD ? currentTargetWord() : mode === MODE_AZ ? currentTargetAZ() : "A";
  const img = alphaImgs[L];

  push();
  fill(255, 220);
  textAlign(CENTER, TOP);
  textSize(11);
  textStyle(BOLD);
  text(headerLabel, x + w / 2, y - 14);

  if (img && img.width > 0 && img.height > 0) {
    const pad = 4;
    const boxW = w - pad * 2;
    const boxH = h - pad * 2;
    const s = Math.min(boxW / img.width, boxH / img.height);
    image(img, x + (w - img.width * s) / 2, y + (h - img.height * s) / 2, img.width * s, img.height * s);
  } else {
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(46);
    textStyle(BOLD);
    text(L, x + w / 2, y + h / 2 + 2);
  }
  pop();
}

function drawPredictionBadgeAt(x, y, w, h) {
  drawCard(x, y, w, h);

  push();
  fill(255, 220);
  textAlign(CENTER, TOP);
  textSize(11);
  textStyle(BOLD);
  text("YOU", x + w / 2, y - 14);
  pop();

  if (!isPredicting || totalExamples() === 0) return;

  const sm = getSmoothedLabel();
  const shown = sm.label && sm.label !== NONE_LABEL ? sm.label : "—";

  push();
  fill(255);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(52);
  text(shown, x + w / 2, y + h / 2 + 2);

  textStyle(NORMAL);
  textSize(12);
  fill(220);
  const c = nf(sm.conf, 1, 2);
  const tag = sm.stable ? "stable" : "…";
  text(`${c} ${tag}`, x + w / 2, y + h - 16);
  pop();
}

function drawCard(x, y, w, h) {
  push();
  noStroke();
  fill(0, 140);
  rect(x + 3, y + 4, w, h, 18);
  fill(20, 20, 24, 210);
  rect(x, y, w, h, 18);
  fill(255, 255, 255, 18);
  rect(x, y, w, 18, 18, 18, 0, 0);
  pop();
}

function drawHoldBar(x, y, w, h, t) {
  push();
  noStroke();
  fill(0, 140);
  rect(x, y, w, h, 999);
  fill(255, 220);
  rect(x, y, w * t, h, 999);
  pop();
}

function drawProgressBar(x, y, w, h, t) {
  push();
  noStroke();
  fill(0, 160);
  rect(x, y, w, h, 999);
  fill(255, 230);
  rect(x, y, w * t, h, 999);
  pop();
}

function drawVignette() {
  push();
  noStroke();
  fill(0, 120);
  rect(0, 0, width, 40);
  fill(0, 90);
  rect(0, height - 50, width, 50);
  fill(0, 60);
  rect(0, 0, 24, height);
  fill(0, 60);
  rect(width - 24, 0, 24, height);
  pop();
}

function drawHUDMinimal(feats) {
  const txt = `ex ${totalExamples()} | hand ${feats ? "yes" : "no"} | ${recordMode ? "REC" : "—"} | mode ${mode}`;

  push();
  noStroke();
  fill(0, 150);
  rect(12, 12, textWidthSafe(txt) + 18, 30, 999);

  fill(255);
  textAlign(LEFT, CENTER);
  textSize(13);
  text(txt, 22, 27);

  if (statusMsg) {
    fill(255, 200);
    textSize(12);
    text(statusMsg, 22, 48);
  }
  pop();
}

function textWidthSafe(s) {
  push();
  textSize(13);
  const w = textWidth(s);
  pop();
  return w;
}

/* ===================== HAND SELECTION ===================== */

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

function getLandmarks21(hand) {
  return hand.keypoints.map((k) => [k.x, k.y, k.z ?? 0]);
}

/* ===================== FEATURES ===================== */

function handToFeatsXYRotNorm(hand) {
  const lm = getLandmarks21(hand);
  const wrist = lm[0];
  const midMcp = lm[9];

  const wx = wrist[0],
    wy = wrist[1];
  const dx = midMcp[0] - wx;
  const dy = midMcp[1] - wy;

  const scale = Math.sqrt(dx * dx + dy * dy) || 1;

  const ang = Math.atan2(dy, dx);
  const rot = -Math.PI / 2 - ang;
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

/* ===================== INPUT ===================== */

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

  if (keyCode === ENTER) {
    if (mode === MODE_AZ) startAZ();
    if (mode === MODE_WORD) {
      const cleaned = sanitizeWord(wordInput?.value?.() ?? wordText);
      if (cleaned) startWord(cleaned);
    }
    return false;
  }

  if (keyCode === RIGHT_ARROW) {
    if (mode === MODE_AZ && azActive && azIdx < 25) advanceAZ(false);
    if (mode === MODE_WORD && wordActive && !wordCompleted && wordIdx < wordText.length - 1) advanceWord(false);
    return false;
  }

  if (keyCode === LEFT_ARROW) {
    if (mode === MODE_AZ && azActive) {
      azIdx = max(0, azIdx - 1);
      correctStartAt = null;
      statusMsg = `Back → ${currentTargetAZ()}`;
    }
    if (mode === MODE_WORD && (wordActive || wordCompleted)) {
      wordIdx = max(0, wordIdx - 1);
      wordCompleted = false;
      wordActive = true;
      correctStartAt = null;
      statusMsg = `Back → ${currentTargetWord()}`;
    }
    return false;
  }

  const kk = key.toUpperCase();
  held[kk] = true;

  if (recordMode) {
    if (LABELS.includes(kk)) {
      recordLabel = kk;
      lastRecordAt = 0;
      statusMsg = `REC ● ${kk}`;
    } else if (kk === "N") {
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

/* ===================== DATASET OPS ===================== */

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
    statusMsg = label === NONE_LABEL ? `Added NONE (${examples[NONE_LABEL].length})` : `Added ${label} (${examples[label].length})`;
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

/* ===================== PERSISTENCE ===================== */

function saveDataset() {
  const payload = {
    version: 9,
    savedAt: new Date().toISOString(),
    examples: examples,
    addHistory: addHistory,
    meta: { feature: "xy_rot_norm_singlehand", trackHand: TRACK_HAND, dims: 42, k: K },
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

/* ===================== EXPORT / IMPORT ===================== */

function exportDataset() {
  const payload = {
    version: 9,
    savedAt: new Date().toISOString(),
    examples: examples,
    addHistory: addHistory,
    meta: { feature: "xy_rot_norm_singlehand", trackHand: TRACK_HAND, dims: 42, k: K },
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

/* ===================== kNN (distance-weighted) + gating ===================== */

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

  if (bestLabel === NONE_LABEL) return { label: NONE_LABEL, conf, best: bestLabel, second: secondLabel, scores };
  if (conf < MIN_CONF || margin < MIN_MARGIN) return { label: NONE_LABEL, conf, best: bestLabel, second: secondLabel, scores };

  return { label: bestLabel, conf, best: bestLabel, second: secondLabel, scores };
}

/* ===================== SMOOTHING ===================== */

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