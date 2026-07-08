import json
import os

DATA_ROOT = "dataset"
classes = sorted(os.listdir(DATA_ROOT))
label_map = {str(i): cls for i, cls in enumerate(classes)}

with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("[*] label_map.json generated!")
