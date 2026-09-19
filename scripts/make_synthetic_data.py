"""
Generates a tiny fake HAM10000-shaped dataset purely so the pipeline code can be
smoke-tested without the real ~2.7GB dataset. Random noise images -- NOT for
training a real model, only for catching bugs in the data/train/compress/benchmark
code path before you plug in the real data on Colab or your own machine.
"""
import os
import csv
import numpy as np
from PIL import Image

OUT_DIR = "/home/claude/edge-ai-skin-lesion/data/HAM10000_synthetic"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGES_PER_CLASS = 6  # small on purpose -- enough for a couple of lesions per class

def main():
    img_dir1 = os.path.join(OUT_DIR, "HAM10000_images_part_1")
    img_dir2 = os.path.join(OUT_DIR, "HAM10000_images_part_2")
    os.makedirs(img_dir1, exist_ok=True)
    os.makedirs(img_dir2, exist_ok=True)

    rows = []
    img_counter = 0
    lesion_counter = 0
    rng = np.random.default_rng(42)

    for cls in CLASSES:
        for lesion_i in range(3):  # 3 lesions per class
            lesion_id = f"HAM_{lesion_counter:05d}"
            lesion_counter += 1
            n_images_this_lesion = 2  # simulate the real dataset's duplicate-image-per-lesion pattern
            for _ in range(n_images_this_lesion):
                image_id = f"ISIC_{img_counter:07d}"
                img_counter += 1
                arr = rng.integers(0, 255, (450, 600, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                target_dir = img_dir1 if img_counter % 2 == 0 else img_dir2
                img.save(os.path.join(target_dir, f"{image_id}.jpg"))
                rows.append({
                    "lesion_id": lesion_id, "image_id": image_id, "dx": cls,
                    "dx_type": "histo", "age": 50, "sex": "male", "localization": "back",
                })

    with open(os.path.join(OUT_DIR, "HAM10000_metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Synthetic dataset written to {OUT_DIR}: {len(rows)} images across {lesion_counter} lesions.")

if __name__ == "__main__":
    main()
