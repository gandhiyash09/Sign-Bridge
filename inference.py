import asyncio
import json
import os
import sys
import tempfile
import threading
import time

import cv2
import edge_tts
# mock tensorflow to bypass mediapipe import issues
sys.modules['tensorflow'] = None
import mediapipe as mp
import numpy as np
import torch
from playsound import playsound

from model import STGCN

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "stgcn_sign_model.pth"
LABEL_MAP_PATH = "label_map.json"

CONF_THRESHOLD = 0.6
COOLDOWN = 1.3

os.makedirs("audio", exist_ok=True)


# -----------------------------------------------------
# TTS stuff
# -----------------------------------------------------
def speak_text(sentence, voice_name):
    def _play():
        output_file = tempfile.mktemp(suffix=".mp3", dir="audio")
        
        async def _tts():
            tts = edge_tts.Communicate(sentence, voice_name)
            await tts.save(output_file)
            
        asyncio.run(_tts())
        playsound(output_file)
        
        try:
            os.remove(output_file)
        except OSError:
            pass

    threading.Thread(target=_play, daemon=True).start()


# -----------------------------------------------------
# load the weights
# -----------------------------------------------------
def load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    # warm-up to init lazy fc
    dummy = torch.zeros(1, 30, 126).to(DEVICE)
    _ = model(dummy)

    model.load_state_dict(checkpoint["model"], strict=False)
    print("[*] model loaded successfully.")

    with open(LABEL_MAP_PATH, "r") as f:
        return json.load(f)


# -----------------------------------------------------
# convert frames to tensor
# -----------------------------------------------------
def format_frames(frames):
    frames_np = []

    for f in frames:
        flat = []
        for p in f:
            flat.extend(p if p is not None else [0.0, 0.0, 0.0])
        frames_np.append(flat)

    arr = np.array(frames_np, dtype=np.float32)

    # Pad or truncate to exactly 30 frames
    if len(arr) < 30:
        pad = np.zeros((30 - len(arr), arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[-30:]

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


# -----------------------------------------------------
# UI drawing functions
# -----------------------------------------------------
def draw_text(img, text, org, font, scale,
                   color_front, color_back,
                   thickness_front, thickness_back):
    # glow / shadow
    cv2.putText(img, text, org, font, scale, color_back,
                thickness_back, cv2.LINE_AA)
    # main text
    cv2.putText(img, text, org, font, scale, color_front,
                thickness_front, cv2.LINE_AA)


def show_text_on_screen(display, sentence, bottom_y, left_x, max_width):
    # draws text at the bottom and wraps it if it gets too long
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Adaptive font scale based on sentence length
    n_chars = len(sentence)
    if n_chars < 35:
        scale = 1.0
    elif n_chars < 80:
        scale = 0.9
    else:
        scale = 0.8

    thickness_front = 2
    thickness_back = 4

    words = sentence.split(" ")
    lines = []
    current_line = ""

    for w in words:
        test_line = (current_line + " " + w).strip()
        (test_w, _), _ = cv2.getTextSize(test_line, font, scale, thickness_front)
        if test_w > max_width and current_line != "":
            lines.append(current_line)
            current_line = w
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    # Keep only last N lines if too many
    max_lines = 3
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    line_height = int(28 * scale)

    # Draw from bottom → up
    for i, line in enumerate(reversed(lines)):
        y = bottom_y - i * line_height
        org = (left_x, y)
        draw_text(
            display,
            line,
            org,
            font,
            scale,
            (255, 255, 255),    # front
            (210, 230, 255),    # glow
            thickness_front,
            thickness_back
        )


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    print(f"[*] running on {DEVICE}")

    # ---------------- Voice selection (f / m) ----------------
    print("\nChoose voice gender for speech:")
    print("  f -> Female (en-US-JennyNeural)")
    print("  m -> Male   (en-US-GuyNeural)")
    choice = input("Enter choice (f/m, default = f): ").strip().lower()

    if choice == "m":
        voice_name = "en-US-GuyNeural"
        print("[*] using male voice")
    else:
        voice_name = "en-US-JennyNeural"
        print("[*] using female voice")

    # ---------------- Label map ----------------
    with open(LABEL_MAP_PATH, "r") as f:
        label_map_raw = json.load(f)

    idx_to_label = {int(k): v for k, v in label_map_raw.items()}
    num_classes = len(idx_to_label)

    # Identify special classes
    def get_idx(target):
        return next((idx for idx, lbl in idx_to_label.items() if lbl.upper() == target), None)

    space_class = get_idx("SPACE")
    end_class = get_idx("END")
    delete_class = get_idx("DELETE")
    hello_class = get_idx("HELLO")
    good_morning_class = get_idx("GOOD MORNING")
    thank_you_class = get_idx("THANK YOU")
    everyone_class = get_idx("EVERYONE")

    print(f"[*] space={space_class}, end={end_class}, delete={delete_class}")
    print(f"[*] hello={hello_class}, good morning={good_morning_class}, "
          f"thank you={thank_you_class}, everyone={everyone_class}")

    # ---------------- Model ----------------
    model = STGCN(in_channels=126, num_classes=num_classes).to(DEVICE)
    load_model(model, MODEL_PATH)
    model.eval()

    # ---------------- Mediapipe Hands ----------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---------------- Camera & Window ----------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[*] error: cannot open camera")
        return

    window_name = "Sign Detection - Tech Blue"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 840)

    print("[*] press ESC to exit.")

    frames_buf = []
    last_time = time.time()

    current_token = "..."
    sentence = ""

    # last_token = last appended letter/word, used for cooldown
    last_token = None
    last_added_time = 0

    fps_time = time.time()
    fps = 0.0

    # ---------------- Main Loop ----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] error: camera failed")
            break

        h, w = frame.shape[:2]
        display = frame.copy()

        # ------------- Keypoint extraction -------------
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        keypoints_frame = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for lm in hand.landmark:
                    keypoints_frame.append([lm.x, lm.y, lm.z])

        if len(keypoints_frame) < 42:
            keypoints_frame += [[0.0, 0.0, 0.0]] * (42 - len(keypoints_frame))
        else:
            keypoints_frame = keypoints_frame[:42]

        frames_buf.append(keypoints_frame)
        if len(frames_buf) > 30:
            frames_buf = frames_buf[-30:]

        # ------------- Inference every 0.7s -------------
        if time.time() - last_time > 0.7:
            x = format_frames(frames_buf).float().to(DEVICE)

            with torch.no_grad():
                preds = model(x)
                probs = torch.softmax(preds, dim=1)

            conf, pred_idx = torch.max(probs, dim=1)
            conf = conf.item()
            pred_idx = pred_idx.item()
            now = time.time()

            if conf < CONF_THRESHOLD:
                current_token = "Not Recognized"

            else:
                label = idx_to_label[pred_idx]
                token_to_add = None

                # ---------- SPACE ----------
                if pred_idx == space_class:
                    token_to_add = " "
                    current_token = "[SPACE]"

                # ---------- END ----------
                elif pred_idx == end_class:
                    current_token = "[END]"
                    if sentence.strip():
                        print(f"[*] sentence: {sentence}")
                        speak_text(sentence, voice_name)
                    sentence = ""
                    last_token = None
                    last_added_time = now

                # ---------- DELETE (undo last char + optional spaces) ----------
                elif pred_idx == delete_class:
                    current_token = "[DELETE]"
                    if sentence:
                        sentence = sentence.rstrip(" ")
                        if sentence:
                            sentence = sentence[:-1]
                    last_token = None
                    last_added_time = now

                # ---------- PHRASE: HELLO ----------
                elif pred_idx == hello_class:
                    token_to_add = "HELLO "
                    current_token = "HELLO"

                # ---------- PHRASE: GOOD MORNING ----------
                elif pred_idx == good_morning_class:
                    token_to_add = "GOOD MORNING "
                    current_token = "GOOD MORNING"

                # ---------- PHRASE: THANK YOU ----------
                elif pred_idx == thank_you_class:
                    token_to_add = "THANK YOU "
                    current_token = "THANK YOU"

                # ---------- PHRASE: EVERYONE ----------
                elif pred_idx == everyone_class:
                    token_to_add = "EVERYONE "
                    current_token = "EVERYONE"

                # ---------- Normal letters (A–Z etc.) ----------
                else:
                    token_to_add = label
                    current_token = label
                
                if token_to_add is not None:
                    if last_token != token_to_add or (now - last_added_time) > COOLDOWN:
                        sentence += token_to_add
                        last_token = token_to_add
                        last_added_time = now

            last_time = time.time()

        # ------------- FPS -------------
        new_time = time.time()
        fps = 1.0 / (new_time - fps_time)
        fps_time = new_time

        # ------------- UI: Tech Blue -------------
        bar_h = 70

        # Top gradient bar
        top_overlay = display.copy()
        start_color = np.array([210, 160, 90], dtype=np.float32)  # BGR-ish blue
        end_color = np.array([160, 120, 70], dtype=np.float32)
        for y in range(bar_h):
            alpha = y / max(bar_h - 1, 1)
            color = (1 - alpha) * start_color + alpha * end_color
            color_tuple = (int(color[0]), int(color[1]), int(color[2]))
            cv2.line(top_overlay, (0, y), (w, y), color_tuple, 1)
        display = cv2.addWeighted(top_overlay, 0.9, display, 0.1, 0)

        # Bottom bar
        bottom_overlay = display.copy()
        cv2.rectangle(bottom_overlay, (0, h - 80), (w, h), (150, 110, 60), -1)
        display = cv2.addWeighted(bottom_overlay, 0.8, display, 0.2, 0)

        # Separator lines
        cv2.line(display, (0, bar_h), (w, bar_h), (255, 220, 150), 2)
        cv2.line(display, (0, h - 80), (w, h - 80), (255, 220, 150), 2)

        # ----- Draw sentence as adaptive multiline -----
        max_text_width = int(w * 0.9)  # 90% of width
        show_text_on_screen(
            display,
            sentence,
            bottom_y=h - 30,
            left_x=25,
            max_width=max_text_width
        )

        # Current token (top-left)
        draw_text(
            display,
            f"Token: {current_token}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            (210, 230, 255),
            2,
            5
        )

        # FPS
        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 220, 255),
            2,
            cv2.LINE_AA
        )

        # Voice info (top-right)
        voice_label = "Voice: Female" if "Jenny" in voice_name else "Voice: Male"
        text_size, _ = cv2.getTextSize(
            voice_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        vx = w - text_size[0] - 25
        vy = 45
        cv2.putText(
            display,
            voice_label,
            (vx, vy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
