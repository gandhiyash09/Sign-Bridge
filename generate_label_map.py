import os, json

data_root = "dataset"
classes = sorted(os.listdir(data_root))
label_map = {str(i): cls for i, cls in enumerate(classes)}

with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("[INFO] label_map.json generated successfully ✅")
