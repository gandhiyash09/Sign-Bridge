import json
import os
import time

import cv2
import mediapipe as mp
import torch

# -----------------------------
# Config
# -----------------------------
SAVE_DIR = "dataset"
RECORD_SECONDS = 1.5
FPS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def record_clip(cap, hands, sign_name):
    """Record a single clip of fixed duration."""
    os.makedirs(os.path.join(SAVE_DIR, sign_name), exist_ok=True)

    print(f"[*] recording '{sign_name}' for {RECORD_SECONDS}s ...")
    frames = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] error: camera failed")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        landmarks = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks = [[None, None, None] for _ in range(21)]

        frames.append(landmarks)

        # Draw hand landmarks and show preview
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Show timer overlay
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Recording {elapsed:.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recorder", frame)

        if elapsed >= RECORD_SECONDS:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pressed during recording
            return False

    # Save the clip
    json_path = os.path.join(SAVE_DIR, sign_name, f"{int(time.time())}.json")
    with open(json_path, "w") as f:
        json.dump({"frames": frames}, f)

    print(f"[*] saved to {json_path} with {len(frames)} frames\n")
    return True


if __name__ == "__main__":
    print(f"[*] running on {DEVICE}")
    print("[*] mediapipe recorder ready\n")

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    sign_name = input("Enter sign name (or 'exit'): ").strip()
    if sign_name.lower() == "exit":
        cap.release()
        cv2.destroyAllWindows()
        exit()

    print("\nControls:")
    print("  SPACE  → Record 1.5s clip")
    print("  ENTER  → Change sign")
    print("  ESC    → Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] error: camera failed")
            break

        cv2.putText(frame, f"Sign: {sign_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "[SPACE]=Record | [ENTER]=Change Sign | [ESC]=Exit",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 13:  # ENTER
            sign_name = input("\nEnter new sign name (or 'exit'): ").strip()
            if sign_name.lower() == "exit":
                break
        elif key == 32:  # SPACE
            success = record_clip(cap, hands, sign_name)
            if not success:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] exited recorder.")
