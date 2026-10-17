# Thanos vs. Human: AI Rock-Paper-Scissors

A real-time Computer Vision and Artificial Intelligence game where a human plays a 5-round Rock-Paper-Scissors match against the Mad Titan, **Thanos**, powered by MediaPipe hand tracking, OpenCV computer vision HUD, and an adaptive Markov Chain AI strategy predictor.

---

## Project Overview & Completed Architecture

| Module | Implementation Details |
| :--- | :--- |
| **Vision & Tracking** | Google MediaPipe Tasks `HandLandmarker` tracking 21 3D joint landmarks at 30+ FPS. |
| **Gesture Classification** | Kinematic finger extension analysis classifying **Rock** (fist), **Paper** (palm), **Scissors** (V-sign), and unpenalized **Unknown** poses. |
| **Thanos AI Brain** | Two-tiered statistical predictor using **First-Order Markov Transitions** $P(\text{Move}_t \mid \text{Move}_{t-1})$ and **Empirical Frequencies**, with random fallback. |
| **Countdown & Lock-In** | Monotonic high-res clock (`time.perf_counter()`) driving a 3, 2, 1, SHOOT! sequence with simultaneous logical lock-in. |
| **Match Engine** | Exactly 5 completed rounds per match, live scoreboard, unpenalized retry logic, and dedicated final victory podium. |
| **Reset Engine** | Instant match reset clearing scores, round counts, and AI history for the next classmate to play without restarting Python. |

---

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the Game:**
   ```bash
   python hand_gesture.py
   ```
3. **Run Automated Test Suite (Deterministic Logic & Scoring):**
   ```bash
   python test_game_logic.py
   ```

---

## Game Flow & State Machine

```
[ LOBBY SCREEN ]
       |  (Click [PLAY] or press SPACE -> resets match & starts Round 1)
       v
[ COUNTDOWN: 3, 2, 1, SHOOT! ]
       |  (At SHOOT!, sample gesture & lock against pre-selected Thanos move)
       +---> Valid Gesture:   Locks moves & resolves round -> [ LOCKED RESULT ]
       +---> Invalid Gesture: Round not counted            -> [ RETRY SCREEN ]
                                                                   | (Click [RETRY])
                                                                   v
                                                              [ COUNTDOWN ]
[ LOCKED RESULT ]
       |
       +---> Rounds 1 to 4: Click [NEXT ROUND] -> [ COUNTDOWN ]
       +---> Round 5:       Click [VIEW FINAL RESULTS]
                                   |
                                   v
                          [ FINAL MATCH OVER PODIUM ]
                                   | (Click [NEW MATCH] or press SPACE)
                                   v
                          [ LOBBY SCREEN ] (Clean Reset for Next Player)
```

---

## Interactive Controls

- **Mouse:** Left-click interactive buttons (`[PLAY]`, `[NEXT ROUND]`, `[RETRY]`, `[NEW MATCH]`, `[< LOBBY]`).
- **SPACE / ENTER:** Universal advance key (Start match $\rightarrow$ Next Round $\rightarrow$ View Results $\rightarrow$ Reset for next player).
- **L:** Return to Lobby (resets match data).
- **Q / ESC:** Cleanly exit application with camera release.
