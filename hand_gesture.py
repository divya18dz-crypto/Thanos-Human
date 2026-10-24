"""
Project: Thanos vs. Human - AI Rock-Paper-Scissors
Step 6: Five-Round Match, Final Victory Screen, and Restart System

Description:
    The complete, production-ready college project application combining:
    1. Real-time webcam hand tracking & gesture recognition (MediaPipe Tasks).
    2. Adaptive Thanos AI Strategy Predictor (Markov Transitions & Move Frequencies).
    3. 3, 2, 1, SHOOT! countdown with simultaneous logical move lock-in.
       - Thanos's counter-move is preselected blindly but remains COMPLETELY HIDDEN
         during countdown and at SHOOT!.
       - Thanos's move is revealed side-by-side with the human's gesture AFTER SHOOT!.
    4. 5-Round Match Flow with continuous live scoreboard across all screens.
       - Scores and completed rounds update strictly after round resolution.
       - Invalid gestures / retries never consume a round or change scores.
    5. Cosmic Thanos HUD theme with Infinity Stone color accents.
    6. Dedicated Final Victory Screen declaring the overall champion.
    7. Clean match reset engine allowing the next classmate to play immediately.

Controls:
    - Mouse Click on [PLAY] / [NEXT ROUND] / [VIEW FINAL RESULTS] / [NEW MATCH] / [< LOBBY]
    - SPACE / ENTER: Universal primary action (Start -> Next Round -> View Results -> Reset)
    - L: Return to Lobby (resets match for a new classmate)
    - Q or ESC: Safely exit application
"""

import os
import sys
import time
import math
import random
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec


# ==============================================================================
# Configuration & Constants
# ==============================================================================
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

CAMERA_INDEX = 0          # Default webcam index (0 for built-in, 1 for external USB)
FRAME_WIDTH = 1280        # Recommended 720p HD resolution
FRAME_HEIGHT = 720
MAX_ROUNDS = 5            # Exactly 5 completed rounds per match

# Countdown timings (in seconds)
COUNTDOWN_STAGE_DURATION = 1.0   # 1 second per count (3, 2, 1)
COUNTDOWN_SHOOT_WINDOW   = 0.8   # 0.8 seconds for SHOOT! sampling window
TOTAL_COUNTDOWN_TIME     = (3 * COUNTDOWN_STAGE_DURATION) + COUNTDOWN_SHOOT_WINDOW

# Counter-move lookup: beats the given move
COUNTER_MOVES = {
    "ROCK": "PAPER",
    "PAPER": "SCISSORS",
    "SCISSORS": "ROCK",
}

# Cosmic Thanos Color Palette (BGR format for OpenCV)
COLOR_VOID_BLACK    = (15, 10, 18)      # Deep obsidian background
COLOR_THANOS_PURPLE = (110, 25, 95)     # Mad Titan deep cosmic purple
COLOR_PURPLE_LIGHT  = (160, 45, 140)    # Illuminated purple for hover
COLOR_INFINITY_GOLD = (40, 215, 255)    # Infinity Gauntlet golden accent
COLOR_GOLD_BRIGHT   = (70, 235, 255)    # Bright gold for button borders
COLOR_CARD_BORDER   = (65, 35, 75)      # Subtle purple card outline
COLOR_TEXT_WHITE    = (255, 255, 255)   # Primary text
COLOR_TEXT_MUTED    = (190, 180, 205)   # Lavender-tinted secondary text
COLOR_MIND_YELLOW   = (40, 220, 255)    # Mind Stone yellow for AI insights

# Gesture Colors
COLOR_ROCK       = (0, 140, 255)        # Coral / Orange
COLOR_PAPER      = (255, 215, 0)        # Sky Blue / Cyan
COLOR_SCISSORS   = (0, 230, 115)        # Bright Lime Green
COLOR_UNKNOWN    = (160, 160, 160)      # Muted Gray
COLOR_NO_HAND    = (60, 60, 230)        # Crimson Red

# Outcome Colors
COLOR_WIN_GREEN  = (0, 230, 115)        # Lime green for Human Win
COLOR_LOSE_RED   = (60, 60, 230)        # Crimson red for Thanos Win
COLOR_DRAW_CYAN  = (255, 215, 0)        # Golden cyan for Draw

GESTURE_COLORS = {
    "ROCK": COLOR_ROCK,
    "PAPER": COLOR_PAPER,
    "SCISSORS": COLOR_SCISSORS,
    "UNKNOWN": COLOR_UNKNOWN,
}

# The Six Infinity Stones (Name, BGR Color)
INFINITY_STONES = [
    ("Power",   (200, 50, 190)),  # Purple
    ("Space",   (240, 130, 30)),  # Blue
    ("Reality", (40, 40, 230)),   # Red
    ("Soul",    (20, 140, 255)),  # Orange
    ("Time",    (50, 210, 80)),   # Green
    ("Mind",    (40, 220, 255)),  # Yellow
]


# ==============================================================================
# Game State Management
# ==============================================================================
class GameState:
    LOBBY         = "LOBBY"          # Start screen with Play button & webcam preview
    COUNTDOWN     = "COUNTDOWN"      # 3, 2, 1, SHOOT! countdown sequence (Thanos hidden)
    LOCKED_RESULT = "LOCKED_RESULT"  # Side-by-side move reveal and round outcome (after SHOOT!)
    RETRY         = "RETRY"          # Prompted when gesture was not detected on SHOOT!
    MATCH_OVER    = "MATCH_OVER"     # Final 5-Round Victory Podium & Restart Engine


# Global mouse interaction state
mouse_pos = [0, 0]
mouse_clicked = False


def on_mouse(event, x, y, flags, param):
    """
    OpenCV mouse event callback tracking cursor coordinates and left clicks.
    """
    global mouse_pos, mouse_clicked
    mouse_pos[0] = x
    mouse_pos[1] = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


# ==============================================================================
# Thanos AI: Strategy Predictor Class
# ==============================================================================
class StrategyPredictor:
    """
    Adaptive statistical prediction engine inspired by the Mind Stone:
    1. First-Order Markov Transition Model: Tracks conditional probabilities P(Move_t | Move_{t-1}).
    2. Empirical Frequency Model: Baseline tracking P(Move) across all completed rounds.
    3. Random Fallback: Used when history is insufficient (e.g. Round 1).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clears all player history for a fresh match."""
        self.history = []
        self.move_counts = {"ROCK": 0, "PAPER": 0, "SCISSORS": 0}
        self.transitions = {
            "ROCK": {"ROCK": 0, "PAPER": 0, "SCISSORS": 0},
            "PAPER": {"ROCK": 0, "PAPER": 0, "SCISSORS": 0},
            "SCISSORS": {"ROCK": 0, "PAPER": 0, "SCISSORS": 0},
        }

    def record_move(self, human_move: str):
        """
        Records a completed, valid human move into history and updates the
        Markov transition matrix. Only called AFTER round resolution.
        """
        if human_move not in COUNTER_MOVES:
            return

        if self.history:
            prev_move = self.history[-1]
            self.transitions[prev_move][human_move] += 1

        self.history.append(human_move)
        self.move_counts[human_move] += 1

    def predict(self) -> dict:
        """
        Forecasts the player's next move and selects the counter-move.
        Must be called BEFORE the round lock-in event using only past rounds.
        """
        # Step 1: Round 1 / Empty History Fallback
        if not self.history:
            pred = random.choice(["ROCK", "PAPER", "SCISSORS"])
            return {
                "predicted_move": pred,
                "counter_move": COUNTER_MOVES[pred],
                "confidence": 0.33,
                "model": "RANDOM_FALLBACK",
                "explanation": "Round 1: Insufficient history. Thanos chose a cosmic random counter.",
            }

        prev_move = self.history[-1]
        trans_counts = self.transitions[prev_move]
        total_trans = sum(trans_counts.values())

        # Step 2: Transition-based Predictor (Markov Chain)
        if total_trans >= 1:
            max_c = max(trans_counts.values())
            candidates = [m for m, c in trans_counts.items() if c == max_c]
            pred = random.choice(candidates)
            conf = max_c / total_trans
            return {
                "predicted_move": pred,
                "counter_move": COUNTER_MOVES[pred],
                "confidence": conf,
                "model": "TRANSITION_MARKOV",
                "explanation": f"Transition Model: After {prev_move}, you played {pred} {max_c}/{total_trans} times ({conf*100:.0f}%).",
            }

        # Step 3: Empirical Frequency Predictor (Baseline)
        total_moves = len(self.history)
        max_c = max(self.move_counts.values())
        candidates = [m for m, c in self.move_counts.items() if c == max_c]
        pred = random.choice(candidates)
        conf = max_c / total_moves
        return {
            "predicted_move": pred,
            "counter_move": COUNTER_MOVES[pred],
            "confidence": conf,
            "model": "OVERALL_FREQUENCY",
            "explanation": f"Frequency Model: {pred} is your most frequent move ({max_c}/{total_moves}, {conf*100:.0f}%).",
        }

    def get_stats(self) -> dict:
        """Returns player history statistics for the Predictability Panel."""
        total = len(self.history)

        def pct(c):
            return f"{(c / total * 100):.0f}%" if total > 0 else "0%"

        r = self.move_counts["ROCK"]
        p = self.move_counts["PAPER"]
        s = self.move_counts["SCISSORS"]
        seq = " -> ".join([m[0] for m in self.history[-6:]]) if self.history else "None"

        return {
            "total": total,
            "rock": f"{r} ({pct(r)})",
            "paper": f"{p} ({pct(p)})",
            "scissors": f"{s} ({pct(s)})",
            "sequence": seq,
        }


# ==============================================================================
# 1. Model Verification & Auto-Download
# ==============================================================================
def ensure_model_exists(model_path: str = MODEL_FILENAME) -> str:
    """
    Checks if the MediaPipe hand_landmarker.task model file is present.
    If missing, automatically downloads it from the official Google repository.
    """
    if os.path.exists(model_path):
        return model_path

    print("=" * 60)
    print(" [i] MediaPipe HandLandmarker model file not found locally.")
    print(f"     Downloading '{model_path}' from Google Storage (~7.8 MB)...")
    print("=" * 60)

    def download_progress(blocks, block_size, total_size):
        if total_size > 0:
            downloaded = blocks * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r     Progress: {percent:5.1f}% [{mb:4.1f} MB / {total_mb:4.1f} MB]", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=download_progress)
        print(f"\n [+] Download completed successfully! Saved to: {os.path.abspath(model_path)}\n")
        return model_path
    except Exception as e:
        print(f"\n\n [!] Error downloading model: {e}")
        print("     Please manually download the model file from:")
        print(f"     {MODEL_URL}")
        print(f"     and place it inside the project folder as: {model_path}\n")
        sys.exit(1)


# ==============================================================================
# 2. Geometric Calculation Helpers
# ==============================================================================
def calculate_distance(lm1, lm2) -> float:
    """Calculates 2D Euclidean distance between two MediaPipe NormalizedLandmarks."""
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)


# ==============================================================================
# 3. Finger State Detection & Gesture Recognition
# ==============================================================================
def detect_finger_states(landmarks) -> dict:
    """
    Determines whether each finger is EXTENDED (True) or FOLDED (False)
    using scale-invariant 3D/2D joint geometry and bone ratio kinematics.
    """
    wrist = landmarks[0]
    palm = landmarks[9]  # Middle MCP joint (palm center reference)
    palm_size = calculate_distance(wrist, palm)
    if palm_size < 1e-4:
        palm_size = 0.1

    finger_indices = {
        "index": (8, 6, 5),
        "middle": (12, 10, 9),
        "ring": (16, 14, 13),
        "pinky": (20, 18, 17),
    }

    states = {}
    for finger_name, (tip_id, pip_id, mcp_id) in finger_indices.items():
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        mcp = landmarks[mcp_id]

        dist_tip_mcp = calculate_distance(tip, mcp)
        dist_pip_mcp = calculate_distance(pip, mcp)
        dist_tip_wrist = calculate_distance(tip, wrist)
        dist_pip_wrist = calculate_distance(pip, wrist)
        dist_tip_palm = calculate_distance(tip, palm)

        ratio_mcp = dist_tip_mcp / max(dist_pip_mcp, 1e-4)
        farther_from_wrist = dist_tip_wrist > (dist_pip_wrist * 1.05)
        far_from_palm = dist_tip_palm > (palm_size * 0.72)
        is_above_pip = tip.y < pip.y

        # Extended if tip projects outward from MCP base bone and wrist/palm
        is_extended = (ratio_mcp > 1.35 and (farther_from_wrist or far_from_palm)) or \
                      (is_above_pip and farther_from_wrist and ratio_mcp > 1.15)
        states[finger_name] = is_extended

    # Thumb: Tip (4), IP (3), MCP (2) relative to Pinky MCP (17) and Index MCP (5)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    pinky_mcp = landmarks[17]

    dist_thumb_pinky = calculate_distance(thumb_tip, pinky_mcp)
    dist_ip_pinky = calculate_distance(thumb_ip, pinky_mcp)
    dist_thumb_mcp = calculate_distance(thumb_tip, thumb_mcp)

    thumb_extended = (dist_thumb_pinky > dist_ip_pinky * 1.05) and (dist_thumb_mcp > palm_size * 0.55)
    states["thumb"] = thumb_extended

    return states


def classify_gesture(finger_states: dict, landmarks) -> tuple:
    """
    Classifies finger states into ROCK, PAPER, SCISSORS, or UNKNOWN.
    Features fast, highly robust fist detection invariant to hand tilt,
    allowing instant recognition of ROCK as soon as fingers fold.
    """
    index = finger_states["index"]
    middle = finger_states["middle"]
    ring = finger_states["ring"]
    pinky = finger_states["pinky"]
    thumb = finger_states["thumb"]

    wrist = landmarks[0]
    palm = landmarks[9]
    palm_size = calculate_distance(wrist, palm)
    if palm_size < 1e-4:
        palm_size = 0.1

    # Compute average tip distance to palm center (landmarks[9])
    tip_indices = [8, 12, 16, 20]
    tip_dists = [calculate_distance(landmarks[i], palm) for i in tip_indices]
    avg_tip_dist = sum(tip_dists) / len(tip_dists)

    extended_count = sum([1 for f in [index, middle, ring, pinky] if f])

    # Reject ONLY genuine deliberate Thumbs Up (thumb sticking high up away from closed fist)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    is_thumbs_up = (
        extended_count == 0 and
        thumb and
        (thumb_tip.y < thumb_mcp.y - palm_size * 0.30) and
        (thumb_tip.y < index_mcp.y - palm_size * 0.30) and
        (calculate_distance(thumb_tip, index_mcp) > palm_size * 0.70)
    )

    if is_thumbs_up:
        return "UNKNOWN", COLOR_UNKNOWN

    # 1. ROCK:
    # High-speed robust fist detection:
    # - All 4 main fingers folded (extended_count == 0)
    # - OR at most 1 finger loose/jittery (extended_count <= 1) while tips are bundled near palm
    # - OR average tip distance to palm is small (compact fist regardless of hand orientation)
    is_rock = (
        (extended_count == 0) or
        (extended_count <= 1 and avg_tip_dist < palm_size * 0.70) or
        (avg_tip_dist < palm_size * 0.62 and not (index and middle))
    )

    if is_rock:
        return "ROCK", COLOR_ROCK

    # 2. PAPER: All 4 main fingers extended (or 3+ with large tip spread)
    if index and middle and ring and pinky:
        return "PAPER", COLOR_PAPER
    if extended_count >= 3 and avg_tip_dist > palm_size * 0.88 and not is_rock:
        return "PAPER", COLOR_PAPER

    # 3. SCISSORS: Index and Middle extended, Ring and Pinky folded
    if index and middle and not ring and not pinky:
        return "SCISSORS", COLOR_SCISSORS
    if index and middle and extended_count == 2:
        return "SCISSORS", COLOR_SCISSORS

    return "UNKNOWN", COLOR_UNKNOWN


# ==============================================================================
# 4. Fair Round Resolution Engine
# ==============================================================================
def determine_winner(human: str, thanos: str) -> tuple:
    """
    Determines the round outcome using standard Rock-Paper-Scissors rules.
    """
    if human == thanos:
        return (
            "DRAW",
            "ROUND DRAW!",
            f"Both champions chose {human}. The timeline remains balanced!",
            COLOR_DRAW_CYAN,
        )

    winning_combos = {
        ("ROCK", "SCISSORS"): "Rock crushes Scissors! The Human strikes a blow.",
        ("SCISSORS", "PAPER"): "Scissors shreds Paper! The Human cuts through destiny.",
        ("PAPER", "ROCK"): "Paper traps Rock! The Human outmaneuvers the Titan.",
    }

    if (human, thanos) in winning_combos:
        return (
            "HUMAN_WINS",
            "HUMAN WINS THIS ROUND!",
            winning_combos[(human, thanos)],
            COLOR_WIN_GREEN,
        )
    else:
        losing_combos = {
            ("SCISSORS", "ROCK"): "Rock crushes Scissors! The Mad Titan asserts dominance.",
            ("PAPER", "SCISSORS"): "Scissors shreds Paper! Thanos cleaves the human attack.",
            ("ROCK", "PAPER"): "Paper traps Rock! Thanos smothers the human resistance.",
        }
        return (
            "THANOS_WINS",
            "THANOS WINS THIS ROUND!",
            losing_combos.get((human, thanos), "Thanos claims this clash."),
            COLOR_LOSE_RED,
        )


# ==============================================================================
# 5. UI Components & Drawing Helpers
# ==============================================================================
def draw_glass_card(img, x: int, y: int, w: int, h: int,
                    bg_color=COLOR_VOID_BLACK, alpha=0.82, border_color=COLOR_CARD_BORDER):
    """
    Draws a translucent dark card on the image with a subtle cosmic border.
    """
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        colored = np.full(roi.shape, bg_color, dtype=np.uint8)
        img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, colored, alpha, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)


def draw_button(img, text: str, x: int, y: int, w: int, h: int,
                is_hovered: bool = False,
                bg_color=COLOR_THANOS_PURPLE, hover_bg=COLOR_PURPLE_LIGHT,
                border_color=COLOR_INFINITY_GOLD, hover_border=COLOR_GOLD_BRIGHT,
                text_color=COLOR_TEXT_WHITE) -> tuple:
    """
    Draws an interactive button with hover illumination and centered text.
    Returns the bounding box (x, y, w, h) for collision testing.
    """
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        fill_color = hover_bg if is_hovered else bg_color
        stroke_color = hover_border if is_hovered else border_color
        alpha = 0.90 if is_hovered else 0.80

        colored = np.full(roi.shape, fill_color, dtype=np.uint8)
        img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, colored, alpha, 0)

        border_thickness = 2 if is_hovered else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), stroke_color, border_thickness, cv2.LINE_AA)

        if is_hovered and x1 > 2 and y1 > 2 and x2 < w_img - 2 and y2 < h_img - 2:
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), stroke_color, 1, cv2.LINE_AA)

        font_scale = 0.75 if len(text) < 14 else 0.60
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, 2, cv2.LINE_AA)

    return (x, y, w, h)


def is_point_inside(px: int, py: int, rect: tuple) -> bool:
    """Checks whether point (px, py) is inside rectangle (x, y, w, h)."""
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def draw_infinity_stones(img, center_x: int, y: int, radius: int = 5, spacing: int = 22):
    """
    Renders the six Infinity Stones as subtle glowing gems in the header.
    """
    total_w = (len(INFINITY_STONES) - 1) * spacing
    start_x = center_x - total_w // 2

    for i, (_, stone_color) in enumerate(INFINITY_STONES):
        cx = start_x + i * spacing
        cv2.circle(img, (cx, y), radius + 2, (10, 8, 15), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, y), radius, stone_color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx - 1, y - 1), max(1, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_header(frame, title: str, subtitle: str, state_badge: str, fps: float,
                round_text: str = "", score_text: str = ""):
    """
    Renders the cosmic header bar with live scoreboard across all screens.
    """
    h, w = frame.shape[:2]
    bar_h = 70
    draw_glass_card(frame, x=0, y=0, w=w, h=bar_h, bg_color=COLOR_VOID_BLACK, alpha=0.90, border_color=COLOR_CARD_BORDER)

    cv2.putText(
        frame, title, (25, 28),
        cv2.FONT_HERSHEY_DUPLEX, 0.78, COLOR_INFINITY_GOLD, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, subtitle, (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA
    )

    # Center: Live Match Scoreboard & Infinity Stones
    if round_text and score_text:
        score_banner = f"{round_text}  |  {score_text}"
        (sw, _), _ = cv2.getTextSize(score_banner, cv2.FONT_HERSHEY_DUPLEX, 0.54, 1)
        sx = (w - sw) // 2
        cv2.putText(frame, score_banner, (sx, 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        draw_infinity_stones(frame, center_x=w // 2, y=52, radius=4, spacing=18)
    else:
        draw_infinity_stones(frame, center_x=w // 2, y=35, radius=5, spacing=20)

    # Right side: State Badge & FPS
    badge_text = f"[{state_badge}]  FPS: {int(fps)}"
    (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
    cv2.putText(
        frame, badge_text, (w - bw - 25, 40),
        cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 230, 150), 1, cv2.LINE_AA
    )


def draw_footer(frame, left_text: str, right_text: str = "Press 'Q' to Quit"):
    """
    Renders the bottom status and shortcut footer.
    """
    h, w = frame.shape[:2]
    footer_h = 36
    draw_glass_card(frame, 0, h - footer_h, w, footer_h, bg_color=COLOR_VOID_BLACK, alpha=0.92, border_color=COLOR_CARD_BORDER)

    cv2.putText(
        frame, left_text, (25, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_MUTED, 1, cv2.LINE_AA
    )

    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(
        frame, right_text, (w - rw - 25, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 160, 180), 1, cv2.LINE_AA
    )


# ==============================================================================
# 6. Screen Renderers
# ==============================================================================
def draw_lobby_screen(frame, hand_detected: bool, fps: float, mx: int, my: int) -> dict:
    """
    Renders the Thanos Lobby Screen with Play button.
    """
    h, w = frame.shape[:2]

    draw_header(
        frame,
        title="THANOS VS. HUMAN",
        subtitle="Can you outsmart the Mad Titan?",
        state_badge="LOBBY",
        fps=fps,
        round_text="MATCH FORMAT: 5 ROUNDS",
        score_text="FIRST TO CLAIM THE UNIVERSE"
    )

    card_w = min(560, int(w * 0.85))
    card_h = 240
    card_x = (w - card_w) // 2
    card_y = 110
    draw_glass_card(frame, card_x, card_y, card_w, card_h, bg_color=COLOR_VOID_BLACK, alpha=0.85, border_color=COLOR_CARD_BORDER)

    cv2.putText(
        frame, "BATTLE PREPARATION",
        (card_x + 30, card_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_INFINITY_GOLD, 2, cv2.LINE_AA
    )

    status_msg = "Hand Detected: Ready to Battle!" if hand_detected else "Show your hand and get ready to battle."
    status_col = (0, 230, 115) if hand_detected else (220, 180, 80)
    dot_col = (0, 255, 0) if hand_detected else (0, 165, 255)

    cv2.circle(frame, (card_x + 35, card_y + 68), 5, dot_col, -1, cv2.LINE_AA)
    cv2.putText(
        frame, status_msg,
        (card_x + 50, card_y + 73), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1, cv2.LINE_AA
    )

    cv2.putText(
        frame, "Combat Moves:",
        (card_x + 30, card_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1, cv2.LINE_AA
    )
    cv2.putText(frame, "* ROCK     : Closed Fist", (card_x + 45, card_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ROCK, 1, cv2.LINE_AA)
    cv2.putText(frame, "* PAPER    : Open Palm", (card_x + 45, card_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PAPER, 1, cv2.LINE_AA)
    cv2.putText(frame, "* SCISSORS : Peace / V-Sign", (card_x + 45, card_y + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SCISSORS, 1, cv2.LINE_AA)

    cv2.putText(
        frame, "Thanos AI Brain: Markov Transition & Move Pattern Predictor Active",
        (card_x + 30, card_y + 218), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MIND_YELLOW, 1, cv2.LINE_AA
    )

    btn_w, btn_h = 240, 58
    btn_x = (w - btn_w) // 2
    btn_y = card_y + card_h + 35
    play_rect = (btn_x, btn_y, btn_w, btn_h)

    is_hovered = is_point_inside(mx, my, play_rect)
    draw_button(
        frame,
        text="[ > PLAY ]",
        x=btn_x, y=btn_y, w=btn_w, h=btn_h,
        is_hovered=is_hovered,
        bg_color=COLOR_THANOS_PURPLE,
        hover_bg=COLOR_PURPLE_LIGHT,
        border_color=COLOR_INFINITY_GOLD,
        hover_border=COLOR_GOLD_BRIGHT,
        text_color=COLOR_TEXT_WHITE
    )

    hint_text = "Click [PLAY] with mouse or press SPACE to start Match"
    (hw, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hint_text, ((w - hw) // 2, btn_y + btn_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 160, 180), 1, cv2.LINE_AA)

    draw_footer(
        frame,
        left_text="Click [PLAY] or Press SPACE to Start  |  Press 'Q' to Quit",
        right_text="Press 'Q' or ESC to Quit"
    )

    return {"play": play_rect}


def draw_countdown_screen(frame, elapsed_time: float, current_gesture: str,
                          gesture_color: tuple, finger_states: dict,
                          hand_detected: bool, rounds_analyzed: int,
                          active_round: int, human_score: int, thanos_score: int, draw_count: int,
                          fps: float, mx: int, my: int) -> dict:
    """
    Renders the live Countdown Screen (3, 2, 1, SHOOT!) with responsive tracking.
    Thanos's move is strictly HIDDEN during this entire countdown sequence.
    """
    h, w = frame.shape[:2]

    round_text = f"ROUND {active_round} / {MAX_ROUNDS}"
    score_text = f"HUMAN: {human_score}  |  THANOS: {thanos_score}  (DRAWS: {draw_count})"

    draw_header(
        frame,
        title="THANOS VS. HUMAN  |  BATTLE ARENA",
        subtitle="Prepare your move! Move will lock at SHOOT!",
        state_badge="COUNTDOWN",
        fps=fps,
        round_text=round_text,
        score_text=score_text
    )

    # 1. Left Telemetry Card (Gesture preview while counting)
    card_w = min(300, int(w * 0.28))
    card_h = 120
    card_x, card_y = 25, 85
    draw_glass_card(frame, card_x, card_y, card_w, card_h, bg_color=COLOR_VOID_BLACK, alpha=0.82)

    cv2.putText(frame, "LIVE HAND TRACKING", (card_x + 16, card_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_INFINITY_GOLD, 1, cv2.LINE_AA)
    display_gesture = current_gesture if hand_detected else "NO HAND"
    display_color = gesture_color if hand_detected else COLOR_NO_HAND
    cv2.putText(frame, display_gesture, (card_x + 16, card_y + 70), cv2.FONT_HERSHEY_DUPLEX, 1.1, display_color, 2, cv2.LINE_AA)

    status_text = "Tracking Active" if hand_detected else "Hand Missing!"
    dot_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.circle(frame, (card_x + 22, card_y + 98), 4, dot_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (card_x + 34, card_y + 103), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # 2. Left Finger Kinematics Panel
    if hand_detected and finger_states:
        tele_w = card_w
        tele_h = 135
        tele_x, tele_y = 25, card_y + card_h + 10
        draw_glass_card(frame, tele_x, tele_y, tele_w, tele_h, bg_color=COLOR_VOID_BLACK, alpha=0.82)
        cv2.putText(frame, "FINGER KINEMATICS", (tele_x + 16, tele_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_INFINITY_GOLD, 1, cv2.LINE_AA)

        finger_list = [
            ("Thumb",  finger_states.get("thumb", False)),
            ("Index",  finger_states.get("index", False)),
            ("Middle", finger_states.get("middle", False)),
            ("Ring",   finger_states.get("ring", False)),
            ("Pinky",  finger_states.get("pinky", False)),
        ]
        row_y = tele_y + 44
        for name, is_up in finger_list:
            state_label = "EXTENDED" if is_up else "FOLDED"
            state_col = (0, 230, 115) if is_up else (140, 140, 150)
            cv2.putText(frame, f"{name}:", (tele_x + 20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, state_label, (tele_x + 95, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, state_col, 1, cv2.LINE_AA)
            row_y += 18

    # 3. Mind Stone Strategy Status (Left Bottom - Thanos's move is HIDDEN)
    ai_w = card_w
    ai_h = 65
    ai_x, ai_y = 25, h - ai_h - 45
    draw_glass_card(frame, ai_x, ai_y, ai_w, ai_h, bg_color=COLOR_VOID_BLACK, alpha=0.82, border_color=(40, 180, 220))
    cv2.putText(frame, "THANOS AI: MIND STONE", (ai_x + 16, ai_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MIND_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Strategy Hidden | Past Rounds: {rounds_analyzed}", (ai_x + 16, ai_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    # 4. Main Central Countdown Visual
    if elapsed_time < 1.0:
        count_label = "3"
        count_color = (220, 50, 190)   # Power Purple
        sub_label = "Get Ready..."
    elif elapsed_time < 2.0:
        count_label = "2"
        count_color = (240, 130, 30)   # Space Blue
        sub_label = "Prepare Move..."
    elif elapsed_time < 3.0:
        count_label = "1"
        count_color = (40, 40, 230)    # Reality Red
        sub_label = "Form Gesture..."
    else:
        count_label = "SHOOT!"
        count_color = COLOR_INFINITY_GOLD  # Golden flash
        sub_label = "LOCKING MOVES NOW!"

    cx, cy = w // 2 + 60, h // 2
    if count_label == "SHOOT!":
        box_w, box_h = 420, 160
        draw_glass_card(frame, cx - box_w // 2, cy - box_h // 2, box_w, box_h, bg_color=COLOR_VOID_BLACK, alpha=0.90, border_color=COLOR_INFINITY_GOLD)
        (tw, th), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_DUPLEX, 2.4, 5)
        cv2.putText(frame, count_label, (cx - tw // 2, cy + 15), cv2.FONT_HERSHEY_DUPLEX, 2.4, count_color, 5, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.putText(frame, sub_label, (cx - sw // 2, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
    else:
        radius = 85
        cv2.circle(frame, (cx, cy), radius + 4, (30, 15, 35), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, count_color, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius - 4, (15, 10, 18), -1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_DUPLEX, 3.2, 6)
        cv2.putText(frame, count_label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX, 3.2, count_color, 6, cv2.LINE_AA)

        (sw, _), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, sub_label, (cx - sw // 2, cy + radius + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    # 5. Top Right [◄ LOBBY] button
    back_w, back_h = 140, 38
    back_x = w - back_w - 25
    back_y = 85
    back_rect = (back_x, back_y, back_w, back_h)

    is_hovered = is_point_inside(mx, my, back_rect)
    draw_button(
        frame,
        text="< LOBBY",
        x=back_x, y=back_y, w=back_w, h=back_h,
        is_hovered=is_hovered,
        bg_color=COLOR_VOID_BLACK,
        hover_bg=COLOR_THANOS_PURPLE,
        border_color=COLOR_CARD_BORDER,
        hover_border=COLOR_INFINITY_GOLD,
        text_color=COLOR_TEXT_WHITE
    )

    draw_footer(
        frame,
        left_text=f"Countdown In Progress: {round_text}  |  Hold move steady on SHOOT!",
        right_text="Press 'Q' or ESC to Quit"
    )

    return {"lobby": back_rect}


def draw_locked_result_screen(frame, human_move: str, thanos_move: str,
                              outcome_code: str, outcome_title: str,
                              outcome_desc: str, outcome_color: tuple,
                              prediction_data: dict, predictor_stats: dict,
                              completed_rounds: int, human_score: int, thanos_score: int, draw_count: int,
                              fps: float, mx: int, my: int) -> dict:
    """
    Renders the Locked Result Screen AFTER SHOOT! featuring:
      - Live updated scoreboard banner
      - Side-by-side move reveal: Human Move vs. Thanos Move
      - Round winner resolution banner
      - The Mind Stone Predictability Panel
      - [▶ NEXT ROUND] (Rounds 1-4) or [▶ VIEW FINAL RESULTS] (Round 5)
    """
    h, w = frame.shape[:2]

    round_text = f"ROUND {completed_rounds} / {MAX_ROUNDS}"
    score_text = f"HUMAN: {human_score}  |  THANOS: {thanos_score}  (DRAWS: {draw_count})"

    draw_header(
        frame,
        title="THANOS VS. HUMAN  |  ROUND CONCLUSION",
        subtitle="Simultaneous Move Reveal & Thanos AI Strategy Insights",
        state_badge="LOCKED RESULT",
        fps=fps,
        round_text=round_text,
        score_text=score_text
    )

    # 1. Top Outcome Banner
    banner_w = min(740, int(w * 0.85))
    banner_h = 75
    banner_x = (w - banner_w) // 2
    banner_y = 80
    draw_glass_card(frame, banner_x, banner_y, banner_w, banner_h, bg_color=COLOR_VOID_BLACK, alpha=0.90, border_color=outcome_color)

    (tw, th), _ = cv2.getTextSize(outcome_title, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    cv2.putText(frame, outcome_title, (banner_x + (banner_w - tw) // 2, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, outcome_color, 2, cv2.LINE_AA)

    (dw, _), _ = cv2.getTextSize(outcome_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(frame, outcome_desc, (banner_x + (banner_w - dw) // 2, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    # 2. Side-by-Side Clash Cards: Revealed AFTER SHOOT!
    card_w = min(360, int(w * 0.38))
    card_h = min(165, int(h * 0.23))
    gap = min(40, int(w * 0.04))
    total_w = card_w * 2 + gap
    left_x = (w - total_w) // 2
    right_x = left_x + card_w + gap
    card_y = banner_y + banner_h + 12

    # Human Card (Left)
    draw_glass_card(frame, left_x, card_y, card_w, card_h, bg_color=COLOR_VOID_BLACK, alpha=0.88, border_color=COLOR_INFINITY_GOLD)
    cv2.putText(frame, "HUMAN FIGHTER", (left_x + 22, card_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    human_col = GESTURE_COLORS.get(human_move, COLOR_TEXT_WHITE)
    cv2.putText(frame, human_move, (left_x + 22, card_y + 80), cv2.FONT_HERSHEY_DUPLEX, 1.3, human_col, 3, cv2.LINE_AA)
    cv2.putText(frame, "[Locked on SHOOT!]", (left_x + 22, card_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 115), 1, cv2.LINE_AA)
    cv2.putText(frame, "Webcam kinematic capture", (left_x + 22, card_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)

    # VS Emblem (Centered between cards)
    vs_cx = left_x + card_w + gap // 2
    vs_cy = card_y + card_h // 2
    cv2.circle(frame, (vs_cx, vs_cy), 24, (30, 15, 35), -1, cv2.LINE_AA)
    cv2.circle(frame, (vs_cx, vs_cy), 22, COLOR_INFINITY_GOLD, 2, cv2.LINE_AA)
    (vsw, vsh), _ = cv2.getTextSize("VS", cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
    cv2.putText(frame, "VS", (vs_cx - vsw // 2, vs_cy + vsh // 2), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_INFINITY_GOLD, 2, cv2.LINE_AA)

    # Thanos Card (Right) - Now Revealed!
    draw_glass_card(frame, right_x, card_y, card_w, card_h, bg_color=COLOR_VOID_BLACK, alpha=0.88, border_color=COLOR_THANOS_PURPLE)
    cv2.putText(frame, "THANOS (MAD TITAN)", (right_x + 22, card_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    thanos_col = GESTURE_COLORS.get(thanos_move, COLOR_TEXT_WHITE)
    cv2.putText(frame, thanos_move, (right_x + 22, card_y + 80), cv2.FONT_HERSHEY_DUPLEX, 1.3, thanos_col, 3, cv2.LINE_AA)
    cv2.putText(frame, f"[Strategy: {prediction_data.get('model', 'AI')}]", (right_x + 22, card_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_INFINITY_GOLD, 1, cv2.LINE_AA)
    cv2.putText(frame, "Preselected before SHOOT! | Revealed now", (right_x + 22, card_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)

    # 3. Predictability Panel
    panel_w = min(820, int(w * 0.88))
    panel_h = min(168, int(h * 0.24))
    panel_x = (w - panel_w) // 2
    panel_y = card_y + card_h + 12
    draw_glass_card(frame, panel_x, panel_y, panel_w, panel_h, bg_color=COLOR_VOID_BLACK, alpha=0.92, border_color=COLOR_MIND_YELLOW)

    cv2.circle(frame, (panel_x + 25, panel_y + 22), 5, COLOR_MIND_YELLOW, -1, cv2.LINE_AA)
    cv2.putText(frame, "THANOS AI: MIND STONE PREDICTABILITY ENGINE", (panel_x + 40, panel_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_MIND_YELLOW, 1, cv2.LINE_AA)

    stats_text = (
        f"Move Frequencies: Rock: {predictor_stats.get('rock', '0')} | "
        f"Paper: {predictor_stats.get('paper', '0')} | "
        f"Scissors: {predictor_stats.get('scissors', '0')} | "
        f"Total Valid Rounds: {predictor_stats.get('total', 0)}"
    )
    cv2.putText(frame, stats_text, (panel_x + 22, panel_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    seq_text = f"Recent Move Sequence (last 6): {predictor_stats.get('sequence', 'None')}"
    cv2.putText(frame, seq_text, (panel_x + 22, panel_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    pred_move = prediction_data.get("predicted_move", "UNKNOWN")
    counter_move = prediction_data.get("counter_move", "UNKNOWN")
    pred_col = GESTURE_COLORS.get(pred_move, COLOR_TEXT_WHITE)
    count_col = GESTURE_COLORS.get(counter_move, COLOR_TEXT_WHITE)

    cv2.putText(frame, "Forecast:", (panel_x + 22, panel_y + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Predicted Human -> {pred_move}", (panel_x + 105, panel_y + 102), cv2.FONT_HERSHEY_DUPLEX, 0.45, pred_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"|  Thanos Counter -> {counter_move}", (panel_x + 360, panel_y + 102), cv2.FONT_HERSHEY_DUPLEX, 0.45, count_col, 1, cv2.LINE_AA)

    reasoning = prediction_data.get("explanation", "Evaluating historical tendencies.")
    cv2.putText(frame, f"Reasoning: {reasoning}", (panel_x + 22, panel_y + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 115), 1, cv2.LINE_AA)

    conf = prediction_data.get("confidence", 0.33) * 100
    disclaimer = f"Confidence: {conf:.0f}%  |  *Empirical probability from limited match history, not a guarantee."
    cv2.putText(frame, disclaimer, (panel_x + 22, panel_y + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 175), 1, cv2.LINE_AA)

    # 4. Action Button: [NEXT ROUND] for Rounds 1-4; [VIEW FINAL RESULTS] for Round 5
    btn_w = min(280, int(w * 0.35))
    btn_h = min(48, int(h * 0.08))
    btn_x = (w - btn_w) // 2
    btn_y = panel_y + panel_h + 12
    action_rect = (btn_x, btn_y, btn_w, btn_h)

    is_action_hovered = is_point_inside(mx, my, action_rect)

    if completed_rounds < MAX_ROUNDS:
        btn_label = f"[ > NEXT ROUND ({completed_rounds + 1}/{MAX_ROUNDS}) ]"
        hint_label = "Click button or press SPACE to begin next round"
    else:
        btn_label = "[ > VIEW FINAL RESULTS ]"
        hint_label = "Match Complete! Click or press SPACE to view overall winner"

    draw_button(
        frame,
        text=btn_label,
        x=btn_x, y=btn_y, w=btn_w, h=btn_h,
        is_hovered=is_action_hovered,
        bg_color=COLOR_THANOS_PURPLE,
        hover_bg=COLOR_PURPLE_LIGHT,
        border_color=COLOR_INFINITY_GOLD,
        hover_border=COLOR_GOLD_BRIGHT,
        text_color=COLOR_TEXT_WHITE
    )

    (hw, _), _ = cv2.getTextSize(hint_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, hint_label, ((w - hw) // 2, btn_y + btn_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 160, 180), 1, cv2.LINE_AA)

    # Top-right [◄ LOBBY] button
    back_w, back_h = 140, 38
    back_x = w - back_w - 25
    back_y = 85
    back_rect = (back_x, back_y, back_w, back_h)
    is_back_hovered = is_point_inside(mx, my, back_rect)
    draw_button(
        frame,
        text="< LOBBY",
        x=back_x, y=back_y, w=back_w, h=back_h,
        is_hovered=is_back_hovered,
        bg_color=COLOR_VOID_BLACK,
        hover_bg=COLOR_THANOS_PURPLE,
        border_color=COLOR_CARD_BORDER,
        hover_border=COLOR_INFINITY_GOLD,
        text_color=COLOR_TEXT_WHITE
    )

    draw_footer(
        frame,
        left_text=f"Round {completed_rounds} of {MAX_ROUNDS} Complete  |  Press SPACE to advance  |  Press 'L' to exit to Lobby",
        right_text="Press 'Q' or ESC to Quit"
    )

    return {"action": action_rect, "lobby": back_rect}


def draw_retry_screen(frame, active_round: int, human_score: int, thanos_score: int, draw_count: int,
                      fps: float, mx: int, my: int) -> dict:
    """
    Renders the unpenalized Retry Screen when no valid gesture was captured on SHOOT!.
    """
    h, w = frame.shape[:2]

    round_text = f"ROUND {active_round} / {MAX_ROUNDS}"
    score_text = f"HUMAN: {human_score}  |  THANOS: {thanos_score}  (DRAWS: {draw_count})"

    draw_header(
        frame,
        title="THANOS VS. HUMAN  |  BATTLE ARENA",
        subtitle="Fair Play System: Gesture Not Detected",
        state_badge="RETRY",
        fps=fps,
        round_text=round_text,
        score_text=score_text
    )

    card_w = min(620, int(w * 0.85))
    card_h = 270
    card_x = (w - card_w) // 2
    card_y = 120
    draw_glass_card(frame, card_x, card_y, card_w, card_h, bg_color=COLOR_VOID_BLACK, alpha=0.90, border_color=(40, 140, 240))

    cv2.putText(frame, "! GESTURE NOT DETECTED", (card_x + 30, card_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Could not confidently classify Rock, Paper, or Scissors on SHOOT!.", (card_x + 30, card_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, "Fair Play Rule: This round was NOT counted into the 5-round match.", (card_x + 30, card_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 115), 1, cv2.LINE_AA)

    cv2.putText(frame, "Tips for successful lock-in:", (card_x + 30, card_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_INFINITY_GOLD, 1, cv2.LINE_AA)
    cv2.putText(frame, "1. Keep your hand upright and directly facing the webcam.", (card_x + 45, card_y + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, "2. Form a clear fist (Rock), open palm (Paper), or V-sign (Scissors).", (card_x + 45, card_y + 196), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, "3. Hold your gesture steady through the SHOOT! call.", (card_x + 45, card_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    btn_w = 260
    btn_h = 56
    btn_x = (w - btn_w) // 2
    btn_y = card_y + card_h + 30
    retry_rect = (btn_x, btn_y, btn_w, btn_h)

    is_retry_hovered = is_point_inside(mx, my, retry_rect)
    draw_button(
        frame,
        text=f"[ RETRY ROUND {active_round} ]",
        x=btn_x, y=btn_y, w=btn_w, h=btn_h,
        is_hovered=is_retry_hovered,
        bg_color=COLOR_THANOS_PURPLE,
        hover_bg=COLOR_PURPLE_LIGHT,
        border_color=COLOR_INFINITY_GOLD,
        hover_border=COLOR_GOLD_BRIGHT,
        text_color=COLOR_TEXT_WHITE
    )

    hint_text = f"Click to retry Round {active_round} or press SPACE"
    (hw, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hint_text, ((w - hw) // 2, btn_y + btn_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 160, 180), 1, cv2.LINE_AA)

    back_w, back_h = 140, 38
    back_x = w - back_w - 25
    back_y = 85
    back_rect = (back_x, back_y, back_w, back_h)
    is_back_hovered = is_point_inside(mx, my, back_rect)
    draw_button(
        frame,
        text="< LOBBY",
        x=back_x, y=back_y, w=back_w, h=back_h,
        is_hovered=is_back_hovered,
        bg_color=COLOR_VOID_BLACK,
        hover_bg=COLOR_THANOS_PURPLE,
        border_color=COLOR_CARD_BORDER,
        hover_border=COLOR_INFINITY_GOLD,
        text_color=COLOR_TEXT_WHITE
    )

    draw_footer(
        frame,
        left_text=f"Round {active_round} Retry: Unpenalized  |  Press SPACE to retry  |  Press 'L' for Lobby",
        right_text="Press 'Q' or ESC to Quit"
    )

    return {"retry": retry_rect, "lobby": back_rect}


def draw_match_over_screen(frame, human_score: int, thanos_score: int, draw_count: int,
                           match_history: list, fps: float, mx: int, my: int) -> dict:
    """
    Renders the Final Victory Screen after Round 5:
      - Overall Match Champion Announcement
      - Dramatic Infinity Stone outcome message
      - Round-by-round recap table
      - Reusable [↺ NEW MATCH / NEXT CLASSMATE] button that resets state completely
    """
    h, w = frame.shape[:2]

    # Determine overall match champion
    if human_score > thanos_score:
        title_banner = "HUMANITY IS SAVED!"
        subtitle_banner = "You outsmarted the Mad Titan with your strategic prowess! The universe is safe."
        outcome_color = COLOR_WIN_GREEN
        trophy_text = f"VICTORY: HUMANITY CLAIMS THE STONES! ({human_score} - {thanos_score})"
    elif thanos_score > human_score:
        title_banner = "THE SNAP IS INEVITABLE!"
        subtitle_banner = "Thanos claims all six Infinity Stones. Half of all life in the universe fades away."
        outcome_color = COLOR_LOSE_RED
        trophy_text = f"DEFEAT: THANOS DOMINATES! ({thanos_score} - {human_score})"
    else:
        title_banner = "PERFECT BALANCE ACHIEVED!"
        subtitle_banner = "Neither champion yielded. The universe rests in cosmic equilibrium."
        outcome_color = COLOR_DRAW_CYAN
        trophy_text = f"DRAW: COSMIC STALEMATE ({human_score} - {thanos_score})"

    draw_header(
        frame,
        title="THANOS VS. HUMAN  |  FINAL MATCH RESULTS",
        subtitle="Match Concluded: Exactly 5 Completed Rounds",
        state_badge="PODIUM",
        fps=fps,
        round_text="MATCH COMPLETE",
        score_text=f"FINAL: HUMAN {human_score} - {thanos_score} THANOS"
    )

    # 1. Main Victory / Defeat Card
    banner_w = min(840, int(w * 0.88))
    banner_h = 100
    banner_x = (w - banner_w) // 2
    banner_y = 82
    draw_glass_card(frame, banner_x, banner_y, banner_w, banner_h, bg_color=COLOR_VOID_BLACK, alpha=0.92, border_color=outcome_color)

    (tw, th), _ = cv2.getTextSize(title_banner, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    cv2.putText(frame, title_banner, (banner_x + (banner_w - tw) // 2, banner_y + 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, outcome_color, 2, cv2.LINE_AA)

    (sw, _), _ = cv2.getTextSize(subtitle_banner, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(frame, subtitle_banner, (banner_x + (banner_w - sw) // 2, banner_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    # 2. Detailed Match Recap Card
    recap_w = banner_w
    recap_h = 240
    recap_x = banner_x
    recap_y = banner_y + banner_h + 12
    draw_glass_card(frame, recap_x, recap_y, recap_w, recap_h, bg_color=COLOR_VOID_BLACK, alpha=0.90, border_color=COLOR_INFINITY_GOLD)

    cv2.putText(frame, trophy_text, (recap_x + 25, recap_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.60, COLOR_INFINITY_GOLD, 2, cv2.LINE_AA)

    score_summary = f"Final Tally:  Human Points: {human_score}   |   Thanos Points: {thanos_score}   |   Ties: {draw_count}"
    cv2.putText(frame, score_summary, (recap_x + 25, recap_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, "Round-by-Round Breakdown:", (recap_x + 25, recap_y + 90), cv2.FONT_HERSHEY_DUPLEX, 0.50, COLOR_MIND_YELLOW, 1, cv2.LINE_AA)

    row_y = recap_y + 115
    for idx, item in enumerate(match_history[:5]):
        r_num = item.get("round", idx + 1)
        h_move = item.get("human", "-")
        t_move = item.get("thanos", "-")
        winner = item.get("winner", "DRAW")

        if winner == "HUMAN_WINS":
            res_str = "HUMAN WIN (+1 pt)"
            res_color = COLOR_WIN_GREEN
        elif winner == "THANOS_WINS":
            res_str = "THANOS WIN (+1 pt)"
            res_color = COLOR_LOSE_RED
        else:
            res_str = "TIE (0 pts)"
            res_color = COLOR_DRAW_CYAN

        log_line = f"Round {r_num}: Human [{h_move}]  vs  Thanos [{t_move}]"
        cv2.putText(frame, log_line, (recap_x + 35, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"--> {res_str}", (recap_x + 400, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, res_color, 1, cv2.LINE_AA)
        row_y += 24

    # 3. Interactive [↺ NEW MATCH / NEXT CLASSMATE] Button
    btn_w = 320
    btn_h = 56
    btn_x = (w - btn_w) // 2
    btn_y = recap_y + recap_h + 16
    restart_rect = (btn_x, btn_y, btn_w, btn_h)

    is_restart_hovered = is_point_inside(mx, my, restart_rect)
    draw_button(
        frame,
        text="[ NEW MATCH / NEXT CLASSMATE ]",
        x=btn_x, y=btn_y, w=btn_w, h=btn_h,
        is_hovered=is_restart_hovered,
        bg_color=COLOR_THANOS_PURPLE,
        hover_bg=COLOR_PURPLE_LIGHT,
        border_color=COLOR_INFINITY_GOLD,
        hover_border=COLOR_GOLD_BRIGHT,
        text_color=COLOR_TEXT_WHITE
    )

    hint_text = "Click or press SPACE to completely reset scores & invite next player"
    (hw, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(frame, hint_text, ((w - hw) // 2, btn_y + btn_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 160, 180), 1, cv2.LINE_AA)

    draw_footer(
        frame,
        left_text="Match Concluded  |  Click [NEW MATCH] or press SPACE to reset for next classmate",
        right_text="Press 'Q' or ESC to Quit"
    )

    return {"restart": restart_rect}


# ==============================================================================
# 7. Main Application Loop
# ==============================================================================
def main():
    global mouse_clicked

    print("=" * 65)
    print(" Thanos vs. Human: Step 6 - 5-Round Match & Final Victory Engine")
    print("=" * 65)

    # 1. Ensure MediaPipe Model Asset Exists
    model_path = ensure_model_exists(MODEL_FILENAME)

    # 2. Configure MediaPipe HandLandmarker Tasks API
    print(" [+] Initializing MediaPipe HandLandmarker (Tasks API)...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 3. Setup Custom Visual Styling for Landmarks
    landmark_style = DrawingSpec(color=COLOR_INFINITY_GOLD, thickness=-1, circle_radius=4)
    connection_style = DrawingSpec(color=COLOR_PURPLE_LIGHT, thickness=2, circle_radius=2)

    # 4. Initialize Strategy Predictor
    predictor = StrategyPredictor()
    print(" [+] Thanos AI Mind Stone Strategy Predictor initialized.")

    # 5. Initialize Webcam
    print(f" [+] Opening webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"\n [!] Error: Unable to open webcam at index {CAMERA_INDEX}.")
        print("     Troubleshooting:")
        print("     1. Verify no other app (Zoom, Teams, Camera) is using the webcam.")
        print("     2. If using an external webcam, try changing CAMERA_INDEX = 1.")
        sys.exit(1)

    window_name = "Thanos vs Human: AI Rock-Paper-Scissors"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n" + "-" * 65)
    print(" Game Engine Complete! Interactive Controls:")
    print("   * Mouse Click on buttons to advance")
    print("   * SPACE / ENTER : Universal shortcut (Start / Next / Results / Reset)")
    print("   * L             : Return to Lobby (resets match for new player)")
    print("   * Q or ESC      : Safely exit application")
    print("-" * 65 + "\n")

    current_state = GameState.LOBBY
    prev_time = time.time()

    # -------------------------------------------------------------
    # 5-Round Match State Variables
    # -------------------------------------------------------------
    completed_rounds = 0       # 0 to 5 completed rounds
    human_score = 0
    thanos_score = 0
    draw_count = 0
    match_history = []         # Records each completed round recap

    countdown_start_time = 0.0
    preselected_thanos_move = "ROCK"
    current_prediction = {}
    locked_human_move = None
    locked_thanos_move = None
    outcome_code = ""
    outcome_title = ""
    outcome_desc = ""
    outcome_color = COLOR_TEXT_WHITE
    last_valid_gesture = None
    last_valid_time = 0.0

    def reset_match():
        """Cleans all match data, scores, rounds, and AI history for the next player."""
        nonlocal completed_rounds, human_score, thanos_score, draw_count, match_history
        nonlocal locked_human_move, locked_thanos_move, outcome_code
        nonlocal outcome_title, outcome_desc, outcome_color, current_state
        nonlocal last_valid_gesture, last_valid_time

        completed_rounds = 0
        human_score = 0
        thanos_score = 0
        draw_count = 0
        match_history = []
        locked_human_move = None
        locked_thanos_move = None
        last_valid_gesture = None
        last_valid_time = 0.0
        outcome_code = ""
        outcome_title = ""
        outcome_desc = ""
        outcome_color = COLOR_TEXT_WHITE
        predictor.reset()
        current_state = GameState.LOBBY
        print(" [Match Reset] Scores, rounds, and AI predictor history wiped cleanly. Ready for next player!")

    def trigger_new_round():
        """
        Initiates a round in the 5-round match:
        - Prevents a 6th round from ever starting.
        - Freezes Thanos's counter-move in secret BEFORE countdown starts.
        - Keeps Thanos's move completely hidden until AFTER SHOOT!.
        """
        nonlocal countdown_start_time, preselected_thanos_move, current_prediction
        nonlocal locked_human_move, locked_thanos_move, outcome_code
        nonlocal outcome_title, outcome_desc, outcome_color, current_state

        # Strict safety check: Never exceed 5 rounds
        if completed_rounds >= MAX_ROUNDS:
            print(" [Match Limit] 5 rounds completed! Routing to MATCH_OVER screen.")
            current_state = GameState.MATCH_OVER
            return

        # Predictor generates forecast and counter-move BEFORE the round starts (kept secret)
        current_prediction = predictor.predict()
        preselected_thanos_move = current_prediction["counter_move"]

        locked_human_move = None
        locked_thanos_move = None
        outcome_code = ""
        outcome_title = ""
        outcome_desc = ""
        outcome_color = COLOR_TEXT_WHITE

        # Start monotonic clock
        countdown_start_time = time.perf_counter()
        current_state = GameState.COUNTDOWN
        active_round = completed_rounds + 1
        print(f"\n [Round {active_round}/{MAX_ROUNDS} Started] Current Score: Human {human_score} - {thanos_score} Thanos (Draws: {draw_count})")
        print(f"                                Thanos AI strategy prepared in secret (hidden from player).")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[!] Warning: Empty frame received from webcam. Exiting loop.")
                break

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = (1.0 / dt) if dt > 0 else 0.0
            prev_time = curr_time

            # Flip horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform real-time hand tracking
            detection_result = detector.detect(mp_image)

            gesture = "UNKNOWN"
            gesture_color = COLOR_UNKNOWN
            finger_states = {}
            hand_detected = False

            if detection_result.hand_landmarks:
                hand_detected = True
                landmarks = detection_result.hand_landmarks[0]

                # Draw skeleton connections and landmarks
                drawing_utils.draw_landmarks(
                    frame,
                    landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style,
                )

                finger_states = detect_finger_states(landmarks)
                gesture, gesture_color = classify_gesture(finger_states, landmarks)

                if gesture in ["ROCK", "PAPER", "SCISSORS"]:
                    last_valid_gesture = gesture
                    last_valid_time = time.perf_counter()

            # -------------------------------------------------------------
            # State Machine & Screen Flow
            # -------------------------------------------------------------
            mx, my = mouse_pos[0], mouse_pos[1]

            if current_state == GameState.LOBBY:
                buttons = draw_lobby_screen(frame, hand_detected, fps, mx, my)
                if mouse_clicked and is_point_inside(mx, my, buttons.get("play", (0, 0, 0, 0))):
                    reset_match()
                    trigger_new_round()
                    mouse_clicked = False

            elif current_state == GameState.COUNTDOWN:
                elapsed = time.perf_counter() - countdown_start_time
                active_round = completed_rounds + 1
                buttons = draw_countdown_screen(
                    frame, elapsed, gesture, gesture_color, finger_states,
                    hand_detected, len(predictor.history),
                    active_round, human_score, thanos_score, draw_count,
                    fps, mx, my
                )

                # 1. At the SHOOT! moment (or entering shoot window), capture and lock the human move immediately
                if elapsed >= 2.8 and locked_human_move is None:
                    if gesture in ["ROCK", "PAPER", "SCISSORS"]:
                        locked_human_move = gesture
                        print(f" [SHOOT! Event] Human move locked: {locked_human_move}. Thanos move remains hidden until reveal.")
                    elif last_valid_gesture and (time.perf_counter() - last_valid_time < 0.8):
                        locked_human_move = last_valid_gesture
                        print(f" [SHOOT! Event] Human move locked from grace buffer: {locked_human_move}.")

                # 2. After SHOOT! finishes (elapsed >= TOTAL_COUNTDOWN_TIME):
                # Now and only now do we reveal Thanos's move, resolve winner, and update scoreboard
                if elapsed >= TOTAL_COUNTDOWN_TIME:
                    if locked_human_move is None and last_valid_gesture and (time.perf_counter() - last_valid_time < 0.8):
                        locked_human_move = last_valid_gesture
                    if locked_human_move is not None:
                        # Reveal Thanos's preselected move
                        locked_thanos_move = preselected_thanos_move

                        # Resolve winner
                        outcome_code, outcome_title, outcome_desc, outcome_color = determine_winner(
                            locked_human_move, locked_thanos_move
                        )

                        # Update score exactly ONCE per completed valid round
                        if outcome_code == "HUMAN_WINS":
                            human_score += 1
                        elif outcome_code == "THANOS_WINS":
                            thanos_score += 1
                        elif outcome_code == "DRAW":
                            draw_count += 1

                        completed_rounds += 1

                        # Record into round recap log
                        match_history.append({
                            "round": completed_rounds,
                            "human": locked_human_move,
                            "thanos": locked_thanos_move,
                            "winner": outcome_code,
                        })

                        # Record valid move into predictor history ONLY AFTER round resolution
                        predictor.record_move(locked_human_move)

                        print(f" [Round {completed_rounds} Resolved] Human: {locked_human_move} vs Thanos: {locked_thanos_move} -> {outcome_code}")
                        print(f"                            Updated Score: Human {human_score} - {thanos_score} Thanos (Draws: {draw_count})")

                        current_state = GameState.LOCKED_RESULT
                    else:
                        print(f" [Lock-In Failed] No valid gesture detected at SHOOT! Round {active_round} entered RETRY.")
                        # Round is not counted; retry will re-attempt the same round
                        current_state = GameState.RETRY

                # Allow canceling countdown back to Lobby
                if mouse_clicked and is_point_inside(mx, my, buttons.get("lobby", (0, 0, 0, 0))):
                    print(" [Event] Countdown aborted by user. Returning to LOBBY.")
                    reset_match()
                    mouse_clicked = False

            elif current_state == GameState.LOCKED_RESULT:
                buttons = draw_locked_result_screen(
                    frame, locked_human_move, locked_thanos_move,
                    outcome_code, outcome_title, outcome_desc, outcome_color,
                    current_prediction, predictor.get_stats(),
                    completed_rounds, human_score, thanos_score, draw_count,
                    fps, mx, my
                )

                # Action button: Advance to next round (Rounds 1-4) OR view final results (Round 5)
                if mouse_clicked and is_point_inside(mx, my, buttons.get("action", (0, 0, 0, 0))):
                    if completed_rounds < MAX_ROUNDS:
                        trigger_new_round()
                    else:
                        print(" [Match Complete] 5 rounds resolved! Advancing to MATCH_OVER podium.")
                        current_state = GameState.MATCH_OVER
                    mouse_clicked = False

                # Lobby button
                elif mouse_clicked and is_point_inside(mx, my, buttons.get("lobby", (0, 0, 0, 0))):
                    reset_match()
                    mouse_clicked = False

            elif current_state == GameState.RETRY:
                # In Retry, active_round reflects the round being re-attempted (completed_rounds + 1)
                active_round = completed_rounds + 1
                buttons = draw_retry_screen(frame, active_round, human_score, thanos_score, draw_count, fps, mx, my)

                # Retry Round button (keeps current match history without penalty)
                if mouse_clicked and is_point_inside(mx, my, buttons.get("retry", (0, 0, 0, 0))):
                    trigger_new_round()
                    mouse_clicked = False

                # Lobby button
                elif mouse_clicked and is_point_inside(mx, my, buttons.get("lobby", (0, 0, 0, 0))):
                    reset_match()
                    mouse_clicked = False

            elif current_state == GameState.MATCH_OVER:
                buttons = draw_match_over_screen(
                    frame, human_score, thanos_score, draw_count, match_history, fps, mx, my
                )

                # Restart button resets all match data and starts fresh for the next classmate
                if mouse_clicked and is_point_inside(mx, my, buttons.get("restart", (0, 0, 0, 0))):
                    reset_match()
                    mouse_clicked = False

            # Reset mouse click at end of frame processing
            mouse_clicked = False

            # Display feed in the OpenCV window
            cv2.imshow(window_name, frame)

            # -------------------------------------------------------------
            # Keyboard Event Handling
            # -------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            # Exit cleanly on 'q', 'Q', or ESC
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\nExit key pressed. Closing application...")
                break

            # SPACE or ENTER: Universal Action Dispatcher
            elif key == ord(' ') or key == 13:
                if current_state == GameState.LOBBY:
                    reset_match()
                    trigger_new_round()
                elif current_state == GameState.LOCKED_RESULT:
                    if completed_rounds < MAX_ROUNDS:
                        trigger_new_round()
                    else:
                        current_state = GameState.MATCH_OVER
                elif current_state == GameState.RETRY:
                    trigger_new_round()
                elif current_state == GameState.MATCH_OVER:
                    reset_match()

            # 'L' returns to Lobby (resets match for new player)
            elif key == ord('l') or key == ord('L'):
                print(" [Keyboard] 'L' pressed -> Returning to LOBBY (match reset).")
                reset_match()

    except Exception as e:
        print(f"\n[!] Unexpected error encountered: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Safe cleanup guaranteed even on unexpected exceptions or keyboard interrupts
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Camera released and all OpenCV windows closed cleanly. Goodbye!\n")


if __name__ == "__main__":
    main()
