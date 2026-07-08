# Sign Language Detection Project

This is our 3rd-semester project for IT-204. It's a real-time sign language detection system that translates hand gestures into text and reads them out loud.

The project uses MediaPipe to extract hand skeletons from a webcam and passes them into a small Graph Convolutional Network (ST-GCN) that we trained in PyTorch. Once the signs are detected, they are concatenated into a sentence. When the `[END]` sign is detected, the program speaks the sentence using edge-tts.

## Team Members
- Yash N Gandhi (241IT082)
- Abhishu Shaurya (241IT003)
- Lavish Temani (241IT040)
- Apurv Rohom (241IT012)

## How to run it

1. Make sure you have python installed.
2. Install the required libraries:
   ```bash
   pip install torch opencv-python numpy mediapipe==0.10.14 edge-tts playsound==1.2.2
   ```
3. Run the main script:
   ```bash
   python inference.py
   ```
4. It will ask you in the terminal to pick a voice gender. Press `Enter` to use the default female voice.
5. The webcam window will open. Show your signs! Press `ESC` to close.

## Files
- `inference.py` - Run this to launch the camera and start translating.
- `model.py` - Contains our PyTorch dataset and model architecture.
- `train_improved.py` - The script we used to train the model.
- `recorder_mediapipe.py` - Small tool we used to record our own signs for the dataset.
- `stgcn_sign_model.pth` - Our trained weights.
- `label_map.json` - Dictionary that maps output IDs to English words.
