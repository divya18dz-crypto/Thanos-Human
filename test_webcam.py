"""
Project: Thanos vs. Human - AI Rock-Paper-Scissors
Step 1: Environment & Webcam Verification Test

Description:
    This script verifies that the required libraries (OpenCV, MediaPipe, NumPy)
    are installed correctly and tests live webcam video capture.
"""

import sys
import time

# -------------------------------------------------------------
# 1. Dependency Verification
# -------------------------------------------------------------
print("=" * 55, flush=True)
print("  Thanos vs. Human: Checking Environment Setup...", flush=True)
print("=" * 55, flush=True)

missing_packages = []

try:
    import cv2
    print(f" [+] OpenCV is installed! Version: {cv2.__version__}", flush=True)
except ImportError:
    missing_packages.append("opencv-python")
    print(" [-] OpenCV is NOT installed.", flush=True)

try:
    import mediapipe as mp
    print(f" [+] MediaPipe is installed! Version: {mp.__version__}", flush=True)
except ImportError:
    missing_packages.append("mediapipe")
    print(" [-] MediaPipe is NOT installed.", flush=True)

try:
    import numpy as np
    print(f" [+] NumPy is installed! Version: {np.__version__}", flush=True)
except ImportError:
    missing_packages.append("numpy")
    print(" [-] NumPy is NOT installed.", flush=True)

if missing_packages:
    print("\n" + "!" * 55, flush=True)
    print(" ERROR: Missing required package(s):", flush=True)
    for pkg in missing_packages:
        print(f"   - {pkg}", flush=True)
    print("\n Run this command in your terminal to install them:", flush=True)
    print(f"   pip install {' '.join(missing_packages)}", flush=True)
    print("!" * 55, flush=True)
    sys.exit(1)

print("\nAll dependencies verified successfully! Starting camera test...\n", flush=True)

# -------------------------------------------------------------
# 2. Webcam Initialization & Video Feed Test
# -------------------------------------------------------------
# 0 is usually the built-in webcam. If using an external USB camera, change to 1 or 2.
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

# Set desired resolution (e.g., 1280x720 HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"[!] Error: Could not open camera at index {CAMERA_INDEX}.")
    print("    - Check if another application (Zoom, Teams, Camera app) is using it.")
    print("    - If using an external camera, try setting CAMERA_INDEX = 1 in the script.")
    sys.exit(1)

print("Camera connected successfully!")
print("Displaying live feed... Press 'q' or 'ESC' in the video window to exit.")

# Variables for FPS calculation
prev_time = 0
window_name = "Thanos vs Human - Camera & Setup Test"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        print("[!] Warning: Failed to read frame from webcam. Exiting loop.")
        break

    # Flip horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    # Calculate real-time FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Add HUD / Status text overlays
    cv2.putText(
        frame,
        "Thanos vs Human AI - Camera Test",
        (20, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Status: System Ready (Press 'q' or ESC to exit)",
        (20, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Show the live frame
    cv2.imshow(window_name, frame)

    # Listen for 'q' or ESC key (key code 27) to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        print("\nExit key pressed. Closing camera feed.")
        break

# -------------------------------------------------------------
# 3. Cleanup Resources
# -------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("Camera released. Environment setup and test completed!\n")
