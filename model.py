# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

# ---------------------
# Adjacency normalization
# ---------------------
def normalize_adj(A, eps=1e-6):
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + eps))
    return D_inv_sqrt @ A @ D_inv_sqrt


# ---------------------
# ST-GCN model (light version)
# ---------------------

class STGCN(nn.Module):
    def __init__(self, num_classes=10, in_channels=126, hidden_dim=128):
        super(STGCN, self).__init__()

        # 1D temporal convs across frames
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # initialize fc lazily (so we can infer its input dim after forward)
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, T, C)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)  # (B, C, T)

        x = self.features(x)    # -> (B, hidden_dim, T)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, hidden_dim)

        # initialize fc dynamically if not set
        if self.fc is None:
            in_dim = x.shape[1]
            self.fc = nn.Linear(in_dim, self.num_classes).to(x.device)

        return self.fc(x)



# ---------------------
# Dataset class
# ---------------------
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.samples = []
        self.label_map = {}
        for i, label in enumerate(sorted(os.listdir(root))):
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue
            self.label_map[label] = i
            for f in os.listdir(label_dir):
                if f.endswith(".json"):
                    self.samples.append((os.path.join(label_dir, f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load JSON
        with open(path, "r") as f:
            data = json.load(f)
        frames = data.get("frames", [])

        clean = []
        for frame in frames:
            # Ensure we have a list of 21 or 42 keypoints (each [x, y, z])
            flat = []

            if isinstance(frame, list):
                for point in frame:
                    if isinstance(point, list) and len(point) >= 3:
                        # Replace None with 0.0
                        flat.extend([
                            point[0] if point[0] is not None else 0.0,
                            point[1] if point[1] is not None else 0.0,
                            point[2] if point[2] is not None else 0.0,
                        ])
                    else:
                        flat.extend([0.0, 0.0, 0.0])
            else:
                # fallback if frame is malformed
                flat = [0.0] * 126

            # fix inconsistent length
            if len(flat) < 126:
                flat += [0.0] * (126 - len(flat))
            elif len(flat) > 126:
                flat = flat[:126]

            clean.append(flat)

        # --- Handle clip length (pad/truncate to fixed 30 frames) ---
        max_frames = 30
        if len(clean) < max_frames:
            clean += [[0.0] * 126 for _ in range(max_frames - len(clean))]
        elif len(clean) > max_frames:
            clean = clean[:max_frames]

        # --- Convert to numpy safely ---
        arr = np.array(clean, dtype=np.float32)
        assert arr.shape == (30, 126), f"Bad shape {arr.shape} in {path}"

        return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)
