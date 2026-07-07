import os
import shutil

# Paths to your dataset folders
datasets = [
    r"C:\Users\lavis\SignLanguage_STGCN\dataset",
    r"C:\Users\lavis\SNS_Project\dataset"
]

# Path to the merged output folder
merged_dataset = r"C:\Users\lavis\PycharmProjects\FinalSNS\dataset"
os.makedirs(merged_dataset, exist_ok=True)

for dataset in datasets:
    for root, dirs, files in os.walk(dataset):
        for file in files:
            src_path = os.path.join(root, file)

            # Determine class name (assumes class is parent folder name)
            class_name = os.path.basename(os.path.dirname(src_path))
            dest_dir = os.path.join(merged_dataset, class_name)
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

print("✅ All datasets merged successfully into:", merged_dataset)
