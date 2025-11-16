import os
from pathlib import Path

ROOT = Path('D:\\Project\\Computer Vision\\FlaskApp-main\\dataset\\Data')
CLASS_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
# ---------------------------------------------

print(f"Scanning {ROOT} for images...")
created = 0

for class_folder in ROOT.iterdir():
    if not class_folder.is_dir():
        continue
    class_name = class_folder.name
    if class_name not in CLASS_MAP:
        print(f"Warning: Skipping unknown folder: {class_name}")
        continue

    class_id = CLASS_MAP[class_name]
    print(f"Processing class: {class_name} (ID: {class_id})")

    for img_path in class_folder.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = img_path.with_suffix(".txt")

        # Skip if already exists
        if label_path.exists():
            continue

        # YOLO format: class_id center_x center_y width height (normalized 0-1)
        label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"

        label_path.write_text(label_content)
        created += 1

print(f"\nDone! Created {created} label files.")
print("   Example: abc.jpg → abc.txt with '2 0.5 0.5 1.0 1.0' (Happy)")
print("   Now run: yolo detect train data=emotion.yaml model=yolov12n.pt cache=False")