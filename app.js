/**
 * Thanos vs. Human: AI Rock-Paper-Scissors (Web Edition)
 * Ported directly from the production Python / OpenCV / MediaPipe implementation.
 * 
 * Features:
 * 1. 21-Joint 3D Hand Tracking via MediaPipe Hands Web SDK.
 * 2. Exact Kinematic Finger State & Gesture Classifier (Rock, Paper, Scissors, Unknown).
 * 3. Thanos Mind Stone AI: First-Order Markov Transition Model & Empirical Frequency.
 * 4. 5-Round Match State Machine with Fair Blind Lock-In & Unpenalized Retries.
 * 5. Procedural Web Audio API sound effects (no external audio assets required).
 */

// ==============================================================================
// 1. Constants & Rules
// ==============================================================================
const COUNTER_MOVES = {
  ROCK: "PAPER",
  PAPER: "SCISSORS",
  SCISSORS: "ROCK",
};

const MOVE_ICONS = {
  ROCK: "✊",
  PAPER: "✋",
  SCISSORS: "✌️",
  UNKNOWN: "❓",
};

const MAX_ROUNDS = 5;

const THANOS_QUOTES = {
  START: [
    '"I will predict your every pattern, mortal. Your choices are bound by destiny."',
    '"Dread it. Run from it. Destiny arrives all the same."',
    '"You could not live with your own failure. Where did that bring you? Back to me."',
  ],
  WIN: [
    '"I am inevitable. You cannot surprise the Mad Titan."',
    '"Reality is often disappointing. Now, reality can be whatever I want."',
    '"You should have gone for the head."',
  ],
  LOSE: [
    '"A commendable strike, mortal. But one battle does not win the cosmos."',
    '"You have my respect, Human. When I am done, half of humanity will still be alive."',
    '"Interesting... you deviated from your established frequency."',
  ],
  DRAW: [
    '"Perfectly balanced, as all things should be."',
    '"A stalemate in the timeline. The cosmos pauses."',
  ],
  MATCH_WIN: '"The universe is balanced once more. Destiny is fulfilled."',
  MATCH_LOSE: '"Impossible... the stones have chosen a new master."',
  MATCH_DRAW: '"A cosmic tie. The multiverse remains in equilibrium."',
};

// ==============================================================================
// 2. Web Audio Synthesis Engine (Procedural Sound FX)
// ==============================================================================
class SoundEngine {
  constructor() {
    this.ctx = null;
    this.muted = false;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  beep(freq, duration, type = "sine", gainVal = 0.15) {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;

    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {
      console.warn("Audio error:", e);
    }
  }

  playCountdown(count) {
    const freqs = { 3: 440, 2: 554, 1: 659 };
    this.beep(freqs[count] || 440, 0.25, "triangle", 0.2);
  }

  playShoot() {
    this.beep(880, 0.4, "sawtooth", 0.25);
    setTimeout(() => this.beep(440, 0.5, "sine", 0.3), 80);
  }

  playWin() {
    const notes = [523.25, 659.25, 783.99, 1046.5]; // C5, E5, G5, C6
    notes.forEach((freq, i) => {
      setTimeout(() => this.beep(freq, 0.3, "triangle", 0.2), i * 90);
    });
  }

  playLose() {
    const notes = [330, 293.66, 261.63, 196];
    notes.forEach((freq, i) => {
      setTimeout(() => this.beep(freq, 0.4, "sawtooth", 0.18), i * 110);
    });
  }

  playDraw() {
    this.beep(523.25, 0.4, "sine", 0.2);
    this.beep(659.25, 0.4, "sine", 0.2);
  }

  playClick() {
    this.beep(800, 0.06, "square", 0.08);
  }
}

const sound = new SoundEngine();

// ==============================================================================
// 3. Thanos AI: Strategy Predictor Class
// ==============================================================================
class StrategyPredictor {
  constructor() {
    this.reset();
  }

  reset() {
    this.history = [];
    this.move_counts = { ROCK: 0, PAPER: 0, SCISSORS: 0 };
    this.transitions = {
      ROCK: { ROCK: 0, PAPER: 0, SCISSORS: 0 },
      PAPER: { ROCK: 0, PAPER: 0, SCISSORS: 0 },
      SCISSORS: { ROCK: 0, PAPER: 0, SCISSORS: 0 },
    };
  }

  recordMove(humanMove) {
    if (!COUNTER_MOVES[humanMove]) return;

    if (this.history.length > 0) {
      const prev = this.history[this.history.length - 1];
      this.transitions[prev][humanMove] += 1;
    }
    this.history.push(humanMove);
    this.move_counts[humanMove] += 1;
  }

  predict() {
    const moves = ["ROCK", "PAPER", "SCISSORS"];

    // Step 1: Round 1 / Empty History Fallback
    if (this.history.length === 0) {
      const pred = moves[Math.floor(Math.random() * moves.length)];
      return {
        predicted_move: pred,
        counter_move: COUNTER_MOVES[pred],
        confidence: 0.33,
        model: "RANDOM_FALLBACK",
        explanation: "Round 1: Insufficient history. Thanos chose a cosmic random counter.",
      };
    }

    const prevMove = this.history[this.history.length - 1];
    const transCounts = this.transitions[prevMove];
    const totalTrans = Object.values(transCounts).reduce((a, b) => a + b, 0);

    // Step 2: Transition-based Predictor (First-Order Markov Chain)
    if (totalTrans >= 1) {
      let maxC = -1;
      let candidates = [];
      for (const [m, c] of Object.entries(transCounts)) {
        if (c > maxC) {
          maxC = c;
          candidates = [m];
        } else if (c === maxC) {
          candidates.push(m);
        }
      }
      const pred = candidates[Math.floor(Math.random() * candidates.length)];
      const conf = maxC / totalTrans;
      return {
        predicted_move: pred,
        counter_move: COUNTER_MOVES[pred],
        confidence: conf,
        model: "TRANSITION_MARKOV",
        explanation: `Markov Model: After ${prevMove}, you chose ${pred} ${maxC}/${totalTrans} times (${Math.round(conf * 100)}%).`,
      };
    }

    // Step 3: Empirical Frequency Predictor (Baseline)
    const totalMoves = this.history.length;
    let maxC = -1;
    let candidates = [];
    for (const [m, c] of Object.entries(this.move_counts)) {
      if (c > maxC) {
        maxC = c;
        candidates = [m];
      } else if (c === maxC) {
        candidates.push(m);
      }
    }
    const pred = candidates[Math.floor(Math.random() * candidates.length)];
    const conf = maxC / totalMoves;
    return {
      predicted_move: pred,
      counter_move: COUNTER_MOVES[pred],
      confidence: conf,
      model: "OVERALL_FREQUENCY",
      explanation: `Frequency Model: ${pred} is your most frequent move (${maxC}/${totalMoves}, ${Math.round(conf * 100)}%).`,
    };
  }

  getStats() {
    const total = this.history.length;
    const pct = (c) => (total > 0 ? `${Math.round((c / total) * 100)}%` : "0%");
    return {
      total,
      rock: `${this.move_counts.ROCK} (${pct(this.move_counts.ROCK)})`,
      paper: `${this.move_counts.PAPER} (${pct(this.move_counts.PAPER)})`,
      scissors: `${this.move_counts.SCISSORS} (${pct(this.move_counts.SCISSORS)})`,
      history: this.history,
    };
  }
}

// ==============================================================================
// 4. Gesture Classification Helpers
// ==============================================================================
function calculateDistance(lm1, lm2) {
  return Math.hypot(lm1.x - lm2.x, lm1.y - lm2.y);
}

function detectFingerStates(landmarks) {
  const wrist = landmarks[0];
  const palm = landmarks[9]; // Middle MCP joint (palm scale reference)
  let palmSize = calculateDistance(wrist, palm);
  if (palmSize < 0.001) palmSize = 0.1;

  const fingerIndices = {
    index: [8, 6, 5],
    middle: [12, 10, 9],
    ring: [16, 14, 13],
    pinky: [20, 18, 17],
  };

  const states = {};

  for (const [fingerName, [tipId, pipId, mcpId]] of Object.entries(fingerIndices)) {
    const tip = landmarks[tipId];
    const pip = landmarks[pipId];
    const mcp = landmarks[mcpId];

    const distTipMcp = calculateDistance(tip, mcp);
    const distPipMcp = calculateDistance(pip, mcp);
    const distTipWrist = calculateDistance(tip, wrist);
    const distPipWrist = calculateDistance(pip, wrist);
    const distTipPalm = calculateDistance(tip, palm);

    const ratioMcp = distTipMcp / Math.max(distPipMcp, 0.0001);
    const fartherFromWrist = distTipWrist > distPipWrist * 1.05;
    const farFromPalm = distTipPalm > palmSize * 0.72;
    const isAbovePip = tip.y < pip.y;

    // Extended if tip clearly projects outward from MCP base bone and wrist/palm
    const isExtended =
      (ratioMcp > 1.35 && (fartherFromWrist || farFromPalm)) ||
      (isAbovePip && fartherFromWrist && ratioMcp > 1.15);

    states[fingerName] = isExtended;
  }

  // Thumb: Tip (4), IP (3), MCP (2) relative to Pinky MCP (17) and Index MCP (5)
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];
  const pinkyMcp = landmarks[17];

  const distThumbPinky = calculateDistance(thumbTip, pinkyMcp);
  const distIpPinky = calculateDistance(thumbIp, pinkyMcp);
  const distThumbMcp = calculateDistance(thumbTip, thumbMcp);

  const thumbExtended =
    distThumbPinky > distIpPinky * 1.05 && distThumbMcp > palmSize * 0.55;
  states.thumb = thumbExtended;

  return states;
}

function classifyGesture(fingerStates, landmarks) {
  const { thumb, index, middle, ring, pinky } = fingerStates;

  const wrist = landmarks[0];
  const palm = landmarks[9];
  let palmSize = calculateDistance(wrist, palm);
  if (palmSize < 0.001) palmSize = 0.1;

  // Compute average tip distance to palm center (landmarks[9])
  const tipIndices = [8, 12, 16, 20];
  const tipDists = tipIndices.map((idx) => calculateDistance(landmarks[idx], palm));
  const avgTipDist = tipDists.reduce((a, b) => a + b, 0) / tipDists.length;

  const extendedCount = [index, middle, ring, pinky].filter(Boolean).length;

  // Reject ONLY genuine deliberate Thumbs Up (thumb sticking high up away from closed fist)
  const thumbTip = landmarks[4];
  const thumbMcp = landmarks[2];
  const indexMcp = landmarks[5];
  const isThumbsUp =
    extendedCount === 0 &&
    thumb &&
    thumbTip.y < thumbMcp.y - palmSize * 0.30 &&
    thumbTip.y < indexMcp.y - palmSize * 0.30 &&
    calculateDistance(thumbTip, indexMcp) > palmSize * 0.70;

  if (isThumbsUp) {
    return "UNKNOWN";
  }

  // 1. ROCK:
  // High-speed robust fist detection:
  // - All 4 main fingers folded (extendedCount === 0)
  // - OR at most 1 finger loose/jittery (extendedCount <= 1) while tips are bundled near palm
  // - OR average tip distance to palm is small (compact fist regardless of hand orientation)
  const isRock =
    extendedCount === 0 ||
    (extendedCount <= 1 && avgTipDist < palmSize * 0.70) ||
    (avgTipDist < palmSize * 0.62 && !(index && middle));

  if (isRock) {
    return "ROCK";
  }

  // 2. PAPER: All 4 main fingers extended (or 3+ with large tip spread)
  if (index && middle && ring && pinky) {
    return "PAPER";
  }
  if (extendedCount >= 3 && avgTipDist > palmSize * 0.88 && !isRock) {
    return "PAPER";
  }

  // 3. SCISSORS: Index and Middle extended, Ring and Pinky folded
  if (index && middle && !ring && !pinky) {
    return "SCISSORS";
  }
  if (index && middle && extendedCount === 2) {
    return "SCISSORS";
  }

  return "UNKNOWN";
}

function determineWinner(human, thanos) {
  if (human === thanos) {
    return {
      outcome: "DRAW",
      title: "ROUND DRAW!",
      desc: `Both champions chose ${human}. The timeline remains balanced!`,
    };
  }

  const winningCombos = {
    "ROCK-SCISSORS": "Rock crushes Scissors! The Human strikes a blow.",
    "SCISSORS-PAPER": "Scissors shreds Paper! The Human cuts through destiny.",
    "PAPER-ROCK": "Paper traps Rock! The Human outmaneuvers the Titan.",
  };

  const key = `${human}-${thanos}`;
  if (winningCombos[key]) {
    return {
      outcome: "HUMAN_WINS",
      title: "HUMAN WINS THIS ROUND!",
      desc: winningCombos[key],
    };
  } else {
    const losingCombos = {
      "SCISSORS-ROCK": "Rock crushes Scissors! The Mad Titan asserts dominance.",
      "PAPER-SCISSORS": "Scissors shreds Paper! Thanos cleaves the human attack.",
      "ROCK-PAPER": "Paper traps Rock! Thanos smothers the human resistance.",
    };
    return {
      outcome: "THANOS_WINS",
      title: "THANOS WINS THIS ROUND!",
      desc: losingCombos[key] || "Thanos claims this clash.",
    };
  }
}

// ==============================================================================
// 5. Main Game Controller
// ==============================================================================
class GameApp {
  constructor() {
    this.ai = new StrategyPredictor();
    this.gameState = "LOBBY"; // LOBBY, COUNTDOWN, LOCKED_RESULT, RETRY, MATCH_OVER

    this.humanScore = 0;
    this.thanosScore = 0;
    this.currentRound = 1; // 1 to 5
    this.roundResults = []; // stores outcomes for pills

    this.liveGesture = "UNKNOWN";
    this.lastValidGesture = null;
    this.lastValidGestureTime = 0;
    this.shootLockedGesture = null;
    this.currentRoundPlan = null; // { predicted_move, counter_move, ... }
    this.lockedHumanMove = null;
    this.lockedThanosMove = null;

    this.cameraActive = false;
    this.handsDetector = null;
    this.cameraInstance = null;

    this.countdownTimer = null;
    this.countdownStage = 0;

    this.initDOMElements();
    this.bindEvents();
    this.updateUI();
  }

  initDOMElements() {
    this.videoEl = document.getElementById("webcam");
    this.canvasEl = document.getElementById("outputCanvas");
    this.ctx = this.canvasEl.getContext("2d");

    this.humanScoreEl = document.getElementById("humanScore");
    this.thanosScoreEl = document.getElementById("thanosScore");
    this.currentRoundEl = document.getElementById("currentRound");
    this.roundPillsEl = document.getElementById("roundPills");
    this.stonesEl = document.getElementById("infinityStones");

    this.liveGestureTextEl = document.getElementById("liveGestureText");
    this.actionBtn = document.getElementById("actionBtn");
    this.actionBtnText = document.getElementById("actionBtnText");
    this.lobbyBtn = document.getElementById("lobbyBtn");

    this.countdownOverlay = document.getElementById("countdownOverlay");
    this.countdownPulse = document.getElementById("countdownPulse");
    this.countdownSubtext = document.getElementById("countdownSubtext");

    this.cameraPrompt = document.getElementById("cameraPrompt");
    this.enableCamBtn = document.getElementById("enableCamBtn");
    this.skipCamBtn = document.getElementById("skipCamBtn");

    this.humanMoveBox = document.getElementById("humanMoveBox");
    this.humanMoveIcon = document.getElementById("humanMoveIcon");
    this.humanMoveName = document.getElementById("humanMoveName");

    this.thanosMoveBox = document.getElementById("thanosMoveBox");
    this.thanosMoveIcon = document.getElementById("thanosMoveIcon");
    this.thanosMoveName = document.getElementById("thanosMoveName");

    this.outcomeTitle = document.getElementById("outcomeTitle");
    this.outcomeDesc = document.getElementById("outcomeDesc");

    this.aiModelBadge = document.getElementById("aiModelBadge");
    this.aiModelName = document.getElementById("aiModelName");
    this.confidenceBar = document.getElementById("confidenceBar");
    this.confidenceVal = document.getElementById("confidenceVal");
    this.aiExplanation = document.getElementById("aiExplanation");
    this.thanosSpeech = document.getElementById("thanosSpeech");

    this.statRock = document.getElementById("statRock");
    this.statPaper = document.getElementById("statPaper");
    this.statScissors = document.getElementById("statScissors");
    this.historyStrip = document.getElementById("historyStrip");

    this.podiumModal = document.getElementById("podiumModal");
    this.podiumIcon = document.getElementById("podiumIcon");
    this.podiumTitle = document.getElementById("podiumTitle");
    this.podiumSubtitle = document.getElementById("podiumSubtitle");
    this.podiumHumanScore = document.getElementById("podiumHumanScore");
    this.podiumThanosScore = document.getElementById("podiumThanosScore");
    this.podiumSummary = document.getElementById("podiumSummary");
    this.podiumRestartBtn = document.getElementById("podiumRestartBtn");

    this.soundBtn = document.getElementById("soundBtn");
    this.soundIcon = document.getElementById("soundIcon");
  }

  bindEvents() {
    this.actionBtn.addEventListener("click", () => this.handlePrimaryAction());
    this.lobbyBtn.addEventListener("click", () => this.resetMatch());
    this.podiumRestartBtn.addEventListener("click", () => this.resetMatch());

    this.enableCamBtn.addEventListener("click", () => this.startCamera());
    this.skipCamBtn.addEventListener("click", () => {
      this.cameraPrompt.style.display = "none";
      this.drawPlaceholderBackground();
    });

    this.soundBtn.addEventListener("click", () => {
      sound.muted = !sound.muted;
      this.soundIcon.textContent = sound.muted ? "🔇" : "🔊";
    });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" || e.code === "Enter") {
        e.preventDefault();
        this.handlePrimaryAction();
      } else if (e.code === "KeyR" && this.gameState === "RETRY") {
        this.startCountdown();
      } else if (e.code === "KeyM") {
        sound.muted = !sound.muted;
        this.soundIcon.textContent = sound.muted ? "🔇" : "🔊";
      } else if (e.code === "KeyL") {
        this.resetMatch();
      }
    });

    // Manual fallback chips
    document.querySelectorAll(".gesture-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        sound.playClick();
        const gesture = chip.dataset.gesture;
        this.liveGesture = gesture;
        this.liveGestureTextEl.textContent = `${gesture} (CLICKED)`;

        if (this.gameState === "COUNTDOWN") {
          this.lockRoundOutcome(gesture);
        } else if (this.gameState === "LOBBY" || this.gameState === "RETRY") {
          // Direct play with this gesture
          this.startCountdown(gesture);
        }
      });
    });
  }

  drawPlaceholderBackground() {
    const canvas = this.canvasEl;
    canvas.width = 1280;
    canvas.height = 720;
    const ctx = this.ctx;

    ctx.fillStyle = "#0e0917";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#ffd166";
    ctx.font = "bold 32px Orbitron, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("MANUAL INTERACTIVE MODE", canvas.width / 2, canvas.height / 2 - 20);

    ctx.fillStyle = "#c7b9e2";
    ctx.font = "20px Outfit, sans-serif";
    ctx.fillText(
      "Use the quick gesture buttons (✊ Rock, ✋ Paper, ✌️ Scissors) below or press Play",
      canvas.width / 2,
      canvas.height / 2 + 25
    );
  }

  // ============================================================================
  // MediaPipe Hands Initialization (Optimized for High-FPS Low-Latency Web)
  // ============================================================================
  async startCamera() {
    sound.init();
    sound.playClick();
    this.cameraPrompt.style.display = "none";

    try {
      this.handsDetector = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      // modelComplexity: 0 uses the lightweight model optimized for browser 60 FPS
      this.handsDetector.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      this.handsDetector.onResults((results) => this.onMediaPipeResults(results));

      // Concurrency guard drops stale frames to eliminate pipeline lag entirely
      let isProcessing = false;
      this.cameraInstance = new Camera(this.videoEl, {
        onFrame: async () => {
          if (isProcessing) return; // Drop frame if previous is still inferring
          isProcessing = true;
          try {
            await this.handsDetector.send({ image: this.videoEl });
          } catch (e) {
            console.warn("MediaPipe processing frame dropped:", e);
          } finally {
            isProcessing = false;
          }
        },
        width: 640,
        height: 480,
      });

      await this.cameraInstance.start();
      this.cameraActive = true;
      this.liveGestureTextEl.textContent = "CAMERA ACTIVE • 60 FPS REAL-TIME";
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Camera access error: " + err.message + "\nYou can still play via Interactive Buttons!");
      this.drawPlaceholderBackground();
    }
  }

  onMediaPipeResults(results) {
    const canvas = this.canvasEl;
    const ctx = this.ctx;

    const w = results.image.width || 640;
    const h = results.image.height || 480;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    // Draw camera image
    ctx.save();
    ctx.drawImage(results.image, 0, 0, w, h);

    // Subtle dark cosmic overlay
    ctx.fillStyle = "rgba(11, 7, 18, 0.18)";
    ctx.fillRect(0, 0, w, h);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];

      // Draw skeleton (zero CPU shadowBlur overhead)
      this.drawSkeleton(landmarks, w, h);

      // Classify Gesture
      const fingerStates = detectFingerStates(landmarks);
      const gesture = classifyGesture(fingerStates, landmarks);
      this.liveGesture = gesture;

      if (gesture !== "UNKNOWN" && gesture !== "NO_HAND") {
        this.lastValidGesture = gesture;
        this.lastValidGestureTime = performance.now();
      }

      const badgeText =
        gesture === "UNKNOWN"
          ? "ANALYZING HAND..."
          : `DETECTED: ${gesture} ${MOVE_ICONS[gesture]}`;
      this.liveGestureTextEl.textContent = badgeText;
    } else {
      this.liveGesture = "NO_HAND";
      this.liveGestureTextEl.textContent = "NO HAND IN FRAME";
    }

    ctx.restore();
  }

  drawSkeleton(landmarks, w, h) {
    const ctx = this.ctx;

    // Hand connections
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // Index
      [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [9, 13], [13, 14], [14, 15], [15, 16],// Ring
      [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
      [0, 17],                              // Palm base
    ];

    // Ultra-fast GPU-friendly drawing (no shadowBlur CPU penalty)
    ctx.strokeStyle = "#bf5af2";
    ctx.lineWidth = 3.5;

    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.stroke();
    });

    // Draw Landmark Points
    landmarks.forEach((p, idx) => {
      ctx.beginPath();
      const isTip = idx === 4 || idx === 8 || idx === 12 || idx === 16 || idx === 20;
      ctx.arc(p.x * w, p.y * h, isTip ? 6 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = idx === 0 ? "#ffd166" : isTip ? "#06d6a0" : "#ffffff";
      ctx.fill();
    });
  }

  // ============================================================================
  // Match & State Machine
  // ============================================================================
  handlePrimaryAction() {
    sound.init();
    sound.playClick();

    switch (this.gameState) {
      case "LOBBY":
        this.startCountdown();
        break;
      case "COUNTDOWN":
        // Premature click locks current live gesture
        if (this.liveGesture !== "NO_HAND" && this.liveGesture !== "UNKNOWN") {
          this.lockRoundOutcome(this.liveGesture);
        }
        break;
      case "LOCKED_RESULT":
        if (this.currentRound < MAX_ROUNDS) {
          this.currentRound += 1;
          this.startCountdown();
        } else {
          this.showFinalPodium();
        }
        break;
      case "RETRY":
        this.startCountdown();
        break;
      case "MATCH_OVER":
        this.resetMatch();
        break;
    }
  }

  startCountdown(overrideGesture = null) {
    if (this.countdownTimer) clearInterval(this.countdownTimer);

    this.gameState = "COUNTDOWN";
    this.lobbyBtn.style.display = "inline-flex";
    this.shootLockedGesture = null;

    // Pre-calculate Thanos's counter-move blindly using past history
    this.currentRoundPlan = this.ai.predict();

    // Reset move boxes to unknown
    this.humanMoveBox.className = "fighter-box";
    this.humanMoveIcon.textContent = "❓";
    this.humanMoveName.textContent = "LOCKING IN...";

    this.thanosMoveBox.className = "fighter-box";
    this.thanosMoveIcon.textContent = "❓";
    this.thanosMoveName.textContent = "COSMIC CHOICE";

    this.outcomeTitle.textContent = "ROUND IN PROGRESS";
    this.outcomeDesc.textContent = "Make your gesture clearly into the camera!";

    // Thanos Quote
    const startQuotes = THANOS_QUOTES.START;
    this.thanosSpeech.textContent = startQuotes[Math.floor(Math.random() * startQuotes.length)];

    // UI Updates
    this.countdownOverlay.style.display = "flex";
    this.actionBtn.disabled = true;
    this.actionBtnText.textContent = "COUNTDOWN...";

    let count = 3;
    this.countdownPulse.textContent = count;
    this.countdownSubtext.textContent = "READY YOUR GESTURE...";
    sound.playCountdown(count);

    // Snappy 750ms interval (2.5s total instead of sluggish 4.5s)
    this.countdownTimer = setInterval(() => {
      count -= 1;
      if (count > 0) {
        this.countdownPulse.textContent = count;
        sound.playCountdown(count);
      } else if (count === 0) {
        this.countdownPulse.textContent = "SHOOT!";
        this.countdownSubtext.textContent = "LOCKING GESTURES!";
        sound.playShoot();
        // Immediately latch valid move at SHOOT!
        if (this.liveGesture && this.liveGesture !== "UNKNOWN" && this.liveGesture !== "NO_HAND") {
          this.shootLockedGesture = this.liveGesture;
        }
      } else {
        clearInterval(this.countdownTimer);
        this.countdownOverlay.style.display = "none";
        this.actionBtn.disabled = false;

        const gestureToLock = overrideGesture || this.shootLockedGesture || this.liveGesture;
        this.lockRoundOutcome(gestureToLock);
      }
    }, 750);
  }

  lockRoundOutcome(humanMove) {
    if (this.countdownTimer) {
      clearInterval(this.countdownTimer);
      this.countdownOverlay.style.display = "none";
      this.actionBtn.disabled = false;
    }

    let finalMove = humanMove;

    if (
      (!finalMove || finalMove === "UNKNOWN" || finalMove === "NO_HAND") &&
      this.shootLockedGesture
    ) {
      finalMove = this.shootLockedGesture;
    }

    // Grace buffer: if live gesture flickered to UNKNOWN right at SHOOT!,
    // use the valid gesture held during the last 800ms!
    if (
      (!finalMove || finalMove === "UNKNOWN" || finalMove === "NO_HAND") &&
      this.lastValidGesture &&
      performance.now() - this.lastValidGestureTime < 800
    ) {
      finalMove = this.lastValidGesture;
    }

    // Check if gesture is valid (Rock, Paper, Scissors)
    if (!finalMove || finalMove === "UNKNOWN" || finalMove === "NO_HAND") {
      this.triggerRetry();
      return;
    }

    this.gameState = "LOCKED_RESULT";
    this.lockedHumanMove = finalMove;
    this.lockedThanosMove = this.currentRoundPlan.counter_move;

    // Determine round winner
    const resolution = determineWinner(this.lockedHumanMove, this.lockedThanosMove);

    // Update Scores
    if (resolution.outcome === "HUMAN_WINS") {
      this.humanScore += 1;
      this.roundResults.push("HUMAN_WINS");
      sound.playWin();
    } else if (resolution.outcome === "THANOS_WINS") {
      this.thanosScore += 1;
      this.roundResults.push("THANOS_WINS");
      sound.playLose();
    } else {
      this.roundResults.push("DRAW");
      sound.playDraw();
    }

    // Record human move into Markov Brain
    this.ai.recordMove(this.lockedHumanMove);

    // Update Clash Arena Cards
    this.humanMoveIcon.textContent = MOVE_ICONS[this.lockedHumanMove];
    this.humanMoveName.textContent = this.lockedHumanMove;

    this.thanosMoveIcon.textContent = MOVE_ICONS[this.lockedThanosMove];
    this.thanosMoveName.textContent = this.lockedThanosMove;

    if (resolution.outcome === "HUMAN_WINS") {
      this.humanMoveBox.className = "fighter-box win";
      this.thanosMoveBox.className = "fighter-box lose";
      this.thanosSpeech.textContent =
        THANOS_QUOTES.LOSE[Math.floor(Math.random() * THANOS_QUOTES.LOSE.length)];
    } else if (resolution.outcome === "THANOS_WINS") {
      this.humanMoveBox.className = "fighter-box lose";
      this.thanosMoveBox.className = "fighter-box win";
      this.thanosSpeech.textContent =
        THANOS_QUOTES.WIN[Math.floor(Math.random() * THANOS_QUOTES.WIN.length)];
    } else {
      this.humanMoveBox.className = "fighter-box draw";
      this.thanosMoveBox.className = "fighter-box draw";
      this.thanosSpeech.textContent =
        THANOS_QUOTES.DRAW[Math.floor(Math.random() * THANOS_QUOTES.DRAW.length)];
    }

    this.outcomeTitle.textContent = resolution.title;
    this.outcomeDesc.textContent = resolution.desc;

    // Button states
    if (this.currentRound < MAX_ROUNDS) {
      this.actionBtnText.textContent = "NEXT ROUND (SPACE)";
    } else {
      this.actionBtnText.textContent = "VIEW FINAL RESULTS (SPACE)";
    }

    this.updateUI();
  }

  triggerRetry() {
    this.gameState = "RETRY";
    sound.playLose();

    this.humanMoveBox.className = "fighter-box lose";
    this.humanMoveIcon.textContent = "❓";
    this.humanMoveName.textContent = "NO GESTURE";

    this.thanosMoveBox.className = "fighter-box";
    this.thanosMoveIcon.textContent = "❓";
    this.thanosMoveName.textContent = "SAVED";

    this.outcomeTitle.textContent = "GESTURE NOT RECOGNIZED";
    this.outcomeDesc.textContent =
      "No penalty applied! Keep your hand clearly in view and click RETRY.";
    this.thanosSpeech.textContent =
      '"Faltering in the face of destiny? Show your move clearly!"';

    this.actionBtnText.textContent = "RETRY ROUND (SPACE)";
  }

  showFinalPodium() {
    this.gameState = "MATCH_OVER";
    this.podiumModal.style.display = "flex";

    this.podiumHumanScore.textContent = this.humanScore;
    this.podiumThanosScore.textContent = this.thanosScore;

    if (this.humanScore > this.thanosScore) {
      this.podiumIcon.textContent = "🏆";
      this.podiumTitle.textContent = "HUMANITY PREVAILS!";
      this.podiumSubtitle = "The Mad Titan has been vanquished.";
      this.podiumSummary.textContent =
        `With a victory of ${this.humanScore} to ${this.thanosScore}, you have outmaneuvered Thanos's predictive AI and saved the cosmos!`;
      this.thanosSpeech.textContent = THANOS_QUOTES.MATCH_LOSE;
      sound.playWin();
    } else if (this.thanosScore > this.humanScore) {
      this.podiumIcon.textContent = "🪐";
      this.podiumTitle.textContent = "THANOS WINS THE CLASH";
      this.podiumSubtitle = "The Snap is completed.";
      this.podiumSummary.textContent =
        `Thanos overpowered humanity ${this.thanosScore} to ${this.humanScore}. His Markov transition matrix anticipated your decisions.`;
      this.thanosSpeech.textContent = THANOS_QUOTES.MATCH_WIN;
      sound.playLose();
    } else {
      this.podiumIcon.textContent = "⚖️";
      this.podiumTitle.textContent = "COSMIC EQUILIBRIUM";
      this.podiumSubtitle = "Draw after 5 rounds.";
      this.podiumSummary.textContent =
        `The clash ended in an even ${this.humanScore}-${this.thanosScore} stalemate. Perfectly balanced, as all things should be.`;
      this.thanosSpeech.textContent = THANOS_QUOTES.MATCH_DRAW;
      sound.playDraw();
    }
  }

  resetMatch() {
    this.ai.reset();
    this.gameState = "LOBBY";
    this.humanScore = 0;
    this.thanosScore = 0;
    this.currentRound = 1;
    this.roundResults = [];
    this.currentRoundPlan = null;
    this.lockedHumanMove = null;
    this.lockedThanosMove = null;
    this.lastValidGesture = null;
    this.lastValidGestureTime = 0;
    this.shootLockedGesture = null;

    if (this.countdownTimer) clearInterval(this.countdownTimer);
    this.countdownOverlay.style.display = "none";
    this.podiumModal.style.display = "none";
    this.lobbyBtn.style.display = "none";

    this.humanMoveBox.className = "fighter-box";
    this.humanMoveIcon.textContent = "❓";
    this.humanMoveName.textContent = "WAITING";

    this.thanosMoveBox.className = "fighter-box";
    this.thanosMoveIcon.textContent = "❓";
    this.thanosMoveName.textContent = "WAITING";

    this.outcomeTitle.textContent = "PRESS PLAY TO BEGIN";
    this.outcomeDesc.textContent = "Five rounds to decide the fate of the universe.";
    this.thanosSpeech.textContent =
      '"A new contender approaches. Show me what moves you possess."';

    this.actionBtn.disabled = false;
    this.actionBtnText.textContent = "PLAY (SPACE)";

    this.updateUI();
  }

  updateUI() {
    this.humanScoreEl.textContent = this.humanScore;
    this.thanosScoreEl.textContent = this.thanosScore;
    this.currentRoundEl.textContent = this.currentRound;

    // Update Round Pills
    for (let r = 1; r <= MAX_ROUNDS; r++) {
      const pill = document.getElementById(`pill-${r}`);
      pill.className = "pill";

      if (r <= this.roundResults.length) {
        const res = this.roundResults[r - 1];
        if (res === "HUMAN_WINS") pill.classList.add("human-won");
        else if (res === "THANOS_WINS") pill.classList.add("thanos-won");
        else pill.classList.add("drawn");
      } else if (r === this.currentRound && this.gameState !== "LOBBY") {
        pill.classList.add("active");
      }
    }

    // Infinity Stones progress
    const stones = this.stonesEl.querySelectorAll(".stone");
    stones.forEach((st, idx) => {
      if (idx < this.roundResults.length) {
        st.classList.add("active");
      } else if (idx === 5) {
        st.classList.add("active"); // Mind stone always glowing
      } else {
        st.classList.remove("active");
      }
    });

    // Predictor Stats
    const stats = this.ai.getStats();
    this.statRock.textContent = stats.rock;
    this.statPaper.textContent = stats.paper;
    this.statScissors.textContent = stats.scissors;

    // AI Prediction Details
    if (this.currentRoundPlan) {
      this.aiModelName.textContent =
        this.currentRoundPlan.model === "TRANSITION_MARKOV"
          ? "Markov Transition Model"
          : this.currentRoundPlan.model === "OVERALL_FREQUENCY"
          ? "Overall Frequency Model"
          : "Random Fallback";

      const pct = Math.round(this.currentRoundPlan.confidence * 100);
      this.confidenceBar.style.width = `${pct}%`;
      this.confidenceVal.textContent = `${pct}%`;
      this.aiExplanation.textContent = this.currentRoundPlan.explanation;
    } else {
      this.confidenceBar.style.width = "33%";
      this.confidenceVal.textContent = "33%";
      this.aiExplanation.textContent =
        "Round 1: Insufficient history. Thanos chose a cosmic random counter.";
    }

    // History Strip
    if (stats.history.length === 0) {
      this.historyStrip.innerHTML = '<span class="empty-text">No completed rounds yet</span>';
    } else {
      this.historyStrip.innerHTML = "";
      stats.history.forEach((m, idx) => {
        const chip = document.createElement("span");
        chip.className = "history-chip";
        chip.textContent = `R${idx + 1}:${MOVE_ICONS[m] || m[0]}`;
        this.historyStrip.appendChild(chip);
      });
    }
  }
}

// ==============================================================================
// 6. Application Bootstrap
// ==============================================================================
window.addEventListener("DOMContentLoaded", () => {
  window.gameApp = new GameApp();
});
