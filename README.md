# Sign-Bridge

This is our 3rd-semester project for IT-204. It's a real-time sign language detection system that translates hand gestures into text and reads them out loud.

The project uses MediaPipe to extract hand skeletons from a webcam and passes them into a small Graph Convolutional Network (ST-GCN) that we trained in PyTorch. Once the signs are detected, they are concatenated into a sentence. When the `[END]` sign is detected, the program speaks the sentence using edge-tts.

## Team Members
- Yash N Gandhi (241IT082)
- Abhishu Shaurya (241IT003)
- Lavish Temani (241IT040)
- Apurv Rohom (241IT012)

## Dataset

We recorded our own dataset from scratch. All four of us took turns in front of the webcam recording short hand gesture clips using `data_tools/recorder_mediapipe.py`. In total we collected around **6000 clips** across all the sign classes (letters A–Z and a few common phrases). The recordings were then merged and used to train the model.

The dataset itself is not in this repo (too large) but can be recreated using the recording script.

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
├── model.py               # ST-GCN architecture and dataset class
├── label_map.json         # Maps class IDs to sign labels
├── stgcn_sign_model.pth   # Trained model weights
├── requirements.txt
│
├── training/              # Scripts used to train the model
│   ├── train_improved.py
│   └── generate_label_map.py
│
└── data_tools/            # Scripts used to build the dataset
    ├── recorder_mediapipe.py
    ├── merger.py
    └── annotator.py
```

## Files

- `inference.py` — Run this to launch the camera and start translating.
- `model.py` — Contains our PyTorch dataset class and ST-GCN model.
- `training/train_improved.py` — The script we used to train the model.
- `training/generate_label_map.py` — Generates `label_map.json` from the dataset folder.
- `data_tools/recorder_mediapipe.py` — Used to record hand gesture clips for the dataset.
- `data_tools/merger.py` — Merges clips from multiple recording sessions into one dataset folder.
- `data_tools/annotator.py` — Small GUI to manually fix bad landmark positions in recorded clips.
- `stgcn_sign_model.pth` — Our trained weights.
- `label_map.json` — Dictionary that maps output IDs to English words.
