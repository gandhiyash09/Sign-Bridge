import torch
import cv2
import numpy as np
import time
from model import STGCN
import mediapipe as mp
import json
import asyncio
import edge_tts
from playsound import playsound
import os

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
# Natural TTS using Edge TTS + playsound
# -----------------------------------------------------
async def speak_text(sentence: str, voice_name: str):
    output_file = "audio/tts_output.mp3"

    tts = edge_tts.Communicate(sentence, voice_name)
    await tts.save(output_file)

    playsound(output_file)

    if os.path.exists(output_file):
        os.remove(output_file)


# -----------------------------------------------------
# Load model checkpoint
# -----------------------------------------------------
def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    # warm-up to init lazy fc
    dummy = torch.zeros(1, 30, 126).to(DEVICE)
    _ = model(dummy)

    model.load_state_dict(checkpoint["model"], strict=False)
    print("[INFO] Model loaded successfully.")

    with open(LABEL_MAP_PATH, "r") as f:
        return json.load(f)


# -----------------------------------------------------
# Preprocess keypoints buffer → tensor
# -----------------------------------------------------
def preprocess_keypoints(frames):
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
# Text drawing helpers (glow + multiline adaptive)
# -----------------------------------------------------
def draw_glow_text(img, text, org, font, scale,
                   color_front, color_back,
                   thickness_front, thickness_back):
    # glow / shadow
    cv2.putText(img, text, org, font, scale, color_back,
                thickness_back, cv2.LINE_AA)
    # main text
    cv2.putText(img, text, org, font, scale, color_front,
                thickness_front, cv2.LINE_AA)


def draw_sentence_multiline(display, sentence, bottom_y, left_x, max_width):
    """
    Draw the sentence in multiple lines at the bottom-left.
    - Wraps words so they don't go out of view.
    - Slightly adapts font size based on length.
    - Keeps only last few lines visible.
    """
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
        draw_glow_text(
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
    print(f"[INFO] Using device: {DEVICE}")

    # ---------------- Voice selection (f / m) ----------------
    print("\nChoose voice gender for speech:")
    print("  f → Female (en-US-JennyNeural)")
    print("  m → Male   (en-US-GuyNeural)")
    choice = input("Enter choice (f/m, default = f): ").strip().lower()

    if choice == "m":
        voice_name = "en-US-GuyNeural"
        print("[INFO] Using MALE voice: en-US-GuyNeural")
    else:
        voice_name = "en-US-JennyNeural"
        print("[INFO] Using FEMALE voice: en-US-JennyNeural")

    # ---------------- Label map ----------------
    with open(LABEL_MAP_PATH, "r") as f:
        label_map_raw = json.load(f)

    idx_to_label = {int(k): v for k, v in label_map_raw.items()}
    num_classes = len(idx_to_label)

    # Identify special classes
    SPACE_CLASS = None
    END_CLASS = None
    DELETE_CLASS = None
    HELLO_CLASS = None
    GOOD_MORNING_CLASS = None
    THANK_YOU_CLASS = None
    EVERYONE_CLASS = None

    for idx, lbl in idx_to_label.items():
        u = lbl.upper()
        if u == "SPACE":
            SPACE_CLASS = idx
        elif u == "END":
            END_CLASS = idx
        elif u == "DELETE":
            DELETE_CLASS = idx
        elif u == "HELLO":
            HELLO_CLASS = idx
        elif u == "GOOD MORNING":
            GOOD_MORNING_CLASS = idx
        elif u == "THANK YOU":
            THANK_YOU_CLASS = idx
        elif u == "EVERYONE":
            EVERYONE_CLASS = idx

    print(f"[INFO] SPACE={SPACE_CLASS}, END={END_CLASS}, DELETE={DELETE_CLASS}")
    print(f"[INFO] HELLO={HELLO_CLASS}, GOOD MORNING={GOOD_MORNING_CLASS}, "
          f"THANK YOU={THANK_YOU_CLASS}, EVERYONE={EVERYONE_CLASS}")

    # ---------------- Model ----------------
    model = STGCN(in_channels=126, num_classes=num_classes).to(DEVICE)
    load_checkpoint(model, MODEL_PATH)
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
        raise RuntimeError("Cannot open camera")

    window_name = "Sign Detection - Tech Blue"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 840)

    print("[INFO] Press ESC to exit.")

    frame_buffer = []
    last_infer_time = time.time()

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

        frame_buffer.append(keypoints_frame)
        if len(frame_buffer) > 30:
            frame_buffer = frame_buffer[-30:]

        # ------------- Inference every 0.7s -------------
        if time.time() - last_infer_time > 0.7:
            x = preprocess_keypoints(frame_buffer).float().to(DEVICE)

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
                token_used = None  # what we actually append / operate on

                # ---------- SPACE ----------
                if pred_idx == SPACE_CLASS:
                    token_used = " "
                    if last_token != token_used or (now - last_added_time) > COOLDOWN:
                        sentence += " "
                        last_token = token_used
                        last_added_time = now
                    current_token = "[SPACE]"

                # ---------- END ----------
                elif pred_idx == END_CLASS:
                    current_token = "[END]"
                    if sentence.strip():
                        print(f"[SENTENCE] {sentence}")
                        asyncio.run(speak_text(sentence, voice_name))
                    sentence = ""
                    last_token = None
                    last_added_time = now

                # ---------- DELETE (undo last char + optional spaces) ----------
                elif pred_idx == DELETE_CLASS:
                    current_token = "[DELETE]"
                    if sentence:
                        # remove trailing spaces
                        while sentence.endswith(" "):
                            sentence = sentence[:-1]
                        # then remove last character
                        if sentence:
                            sentence = sentence[:-1]
                    last_token = None
                    last_added_time = now

                # ---------- PHRASE: HELLO ----------
                elif pred_idx == HELLO_CLASS:
                    token_used = "HELLO"
                    if last_token != token_used or (now - last_added_time) > COOLDOWN:
                        sentence += "HELLO "
                        last_token = token_used
                        last_added_time = now
                    current_token = "HELLO"

                # ---------- PHRASE: GOOD MORNING ----------
                elif pred_idx == GOOD_MORNING_CLASS:
                    token_used = "GOOD MORNING"
                    if last_token != token_used or (now - last_added_time) > COOLDOWN:
                        sentence += "GOOD MORNING "
                        last_token = token_used
                        last_added_time = now
                    current_token = "GOOD MORNING"

                # ---------- PHRASE: THANK YOU ----------
                elif pred_idx == THANK_YOU_CLASS:
                    token_used = "THANK YOU"
                    if last_token != token_used or (now - last_added_time) > COOLDOWN:
                        sentence += "THANK YOU "
                        last_token = token_used
                        last_added_time = now
                    current_token = "THANK YOU"

                # ---------- PHRASE: EVERYONE ----------
                elif pred_idx == EVERYONE_CLASS:
                    token_used = "EVERYONE"
                    if last_token != token_used or (now - last_added_time) > COOLDOWN:
                        sentence += "EVERYONE "
                        last_token = token_used
                        last_added_time = now
                    current_token = "EVERYONE"

                # ---------- Normal letters (A–Z etc.) ----------
                else:
                    token_used = label  # usually a single letter
                    if last_token != token_used or (now - last_added_time) > COOLDOWN:
                        sentence += label
                        last_token = token_used
                        last_added_time = now
                    current_token = label

            last_infer_time = time.time()

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
        draw_sentence_multiline(
            display,
            sentence,
            bottom_y=h - 30,
            left_x=25,
            max_width=max_text_width
        )

        # Current token (top-left)
        draw_glow_text(
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
