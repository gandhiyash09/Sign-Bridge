# Sign-Bridge

A real-time sign language translator: show a hand gesture to your webcam and it recognizes the sign, builds a sentence out of consecutive signs, and reads it out loud.

Built as our 3rd-semester project for IT-204.

## How it works

1. **MediaPipe Hands** extracts up to two hands' worth of 21 skeletal landmarks (x, y, z) per frame from the webcam feed.
2. Landmarks are buffered into a rolling 30-frame window and re-classified roughly every 0.7 seconds.
3. A custom **1D temporal convolutional network (TCN)**, trained from scratch in PyTorch, classifies the window into one of 33 gesture classes.
4. Predictions below a confidence threshold are discarded, and a short cooldown stops one held gesture from being added to the sentence over and over.
5. `SPACE`, `DELETE`, and `END` are treated as editing commands rather than letters. `HELLO`, `GOOD MORNING`, `THANK YOU`, and `EVERYONE` are recognized as single full-word gestures.
6. On `END`, the finished sentence is spoken aloud using `edge-tts`, running on a background thread so the camera feed never freezes while audio plays.

## Team Members
- Yash N Gandhi (241IT082)
- Abhishu Shaurya (241IT003)
- Lavish Temani (241IT040)
- Apurv Rohom (241IT012)

## Dataset

We recorded our own dataset from scratch — all four of us took turns in front of the webcam recording short gesture clips with `data_tools/recorder_mediapipe.py`. In total we collected **6,043 clips across 33 classes**: the letters A–Z, plus SPACE, DELETE, END, HELLO, GOOD MORNING, THANK YOU, and EVERYONE.

Each clip is a JSON file of per-frame hand landmarks (`{"frames": [...]}`) — one file per recording. Since each of us recorded separately, `data_tools/merger.py` combines everyone's session folders into a single `dataset/` directory and auto-renames any filename collisions. `data_tools/annotator.py` is a small GUI we used afterward to manually drag-correct any frames where MediaPipe mislocated a landmark.

The dataset itself isn't in this repo (too large), but can be recreated with the recording script above.

## How to run it

1. Make sure you have Python installed.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python inference.py
   ```
4. It will ask you in the terminal to pick a voice gender. Press `Enter` to use the default female voice.
5. The webcam window will open. Show your signs! Press `ESC` to close.

## Project Structure

```
Sign-Bridge/
├── inference.py           # Run this to start the translator
├── model.py               # TCN architecture and dataset class
├── label_map.json         # Maps class IDs to sign labels
├── stgcn_sign_model.pth   # Trained model weights
├── requirements.txt
│
├── training/               # Scripts used to train the model
│   ├── train_improved.py
│   └── generate_label_map.py
│
└── data_tools/              # Scripts used to build the dataset
    ├── recorder_mediapipe.py
    ├── merger.py
    └── annotator.py
```

## Files

- `inference.py` — Run this to launch the camera and start translating.
- `model.py` — Contains our PyTorch dataset class and TCN model.
- `training/train_improved.py` — The script we used to train the model.
- `training/generate_label_map.py` — Generates `label_map.json` from the dataset folder.
- `data_tools/recorder_mediapipe.py` — Used to record hand gesture clips for the dataset.
- `data_tools/merger.py` — Merges clips from multiple recording sessions into one dataset folder.
- `data_tools/annotator.py` — Small GUI to manually fix bad landmark positions in recorded clips.
- `stgcn_sign_model.pth` — Our trained weights.
- `label_map.json` — Dictionary that maps output IDs to English words.

## Possible improvements

- Hold out a proper train/validation split before training, so classification accuracy can actually be measured on unseen clips instead of only tracking training loss.
- Extend the temporal model into a true spatial-temporal graph convolution over the hand's joint connectivity — the groundwork for an adjacency matrix already exists as a utility in `model.py`, it just isn't wired into the forward pass yet.
- Handle two-hand simultaneous signs as compound gestures, rather than concatenating both hands' landmarks into one classification window.
