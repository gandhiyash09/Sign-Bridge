import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import STGCN, SignDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
LABEL_MAP_PATH = "label_map.json"
SAVE_PATH = "stgcn_sign_model.pth"
EPOCHS = 200           # higher = more accuracy
BATCH_SIZE = 4
LR = 1e-4


def load_label_map():
    if not os.path.exists(LABEL_MAP_PATH):
        print("[*] error: label_map.json missing")
        return None

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    # keys become integers
    label_map = {int(k): v for k, v in label_map.items()}

    print(f"[*] loaded {len(label_map)} classes from label map")
    return label_map


def train():
    label_map = load_label_map()
    if label_map is None: return

    num_classes = len(label_map)

    dataset = SignDataset(DATASET_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = STGCN(in_channels=126, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = float("inf")

    print("[*] starting training...")
    print(f"[*] training on {len(dataset)} samples, {num_classes} classes.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss

            checkpoint = {
                "model": model.state_dict(),
                "label_map": label_map
            }

            torch.save(checkpoint, SAVE_PATH)
            print("[*] saved new best model!")

    print("[*] done training.")


if __name__ == "__main__":
    train()
