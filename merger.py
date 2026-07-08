import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = [
    PROJECT_ROOT.parent / "SignLanguage_STGCN" / "dataset",
    PROJECT_ROOT.parent / "SNS_Project" / "dataset"
]

MERGED_DATASET = PROJECT_ROOT / "dataset"
os.makedirs(MERGED_DATASET, exist_ok=True)

for dataset in DATASETS:
    for root, dirs, files in os.walk(dataset):
        for file in files:
            src_path = os.path.join(root, file)

            # Determine class name (assumes class is parent folder name)
            class_name = os.path.basename(os.path.dirname(src_path))
            dest_dir = os.path.join(MERGED_DATASET, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            # Create unique filename if duplicate exists
            filename = os.path.basename(file)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1

            shutil.copy2(src_path, dest_path)

print(f"[*] merged everything into: {MERGED_DATASET}")
