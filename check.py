import os, cv2
from PIL import Image

def check_folder(folder):
    bad = []
    for f in sorted(os.listdir(folder)):
        path = os.path.join(folder, f)

        # 1. tiny or empty file
        if os.path.getsize(path) < 1024:
            bad.append((f, "file too small"))
            continue

        # 2. PIL integrity check (catches truncated files)
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as e:
            bad.append((f, f"PIL verify failed: {e}"))
            continue

        # 3. OpenCV load check
        img = cv2.imread(path)
        if img is None:
            bad.append((f, "cv2 could not read"))
            continue

        # 4. channels
        if img.ndim != 3 or img.shape[2] != 3:
            bad.append((f, f"unexpected shape {img.shape}"))
            continue

        # 5. blank image
        if img.std() < 2:
            bad.append((f, "almost blank"))

    return bad

for folder in ["data/raw-890", "data/reference-890"]:
    problems = check_folder(folder)
    print(folder, "->", len(problems), "problems")
    for p in problems:
        print("  ", p)