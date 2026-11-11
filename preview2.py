import random
import argparse
from pathlib import Path

import cv2
import numpy as np

from dip.scene_adaptive import enhance_scene_adaptive

PANEL = 256


def fit(img):
    h, w = img.shape[:2]
    scale = PANEL / max(h, w)

    img = cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
    y = (PANEL - img.shape[0]) // 2
    x = (PANEL - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img

    return canvas


def label(img, title):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (PANEL, 22), (0, 0, 0), -1)
    cv2.putText(
        img, title, (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    raw_dir = Path("data/raw-890")
    images = [
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not images:
        raise SystemExit("No images found in data/raw-890")

    selected = random.sample(images, min(args.n, len(images)))

    for path in selected:
        print("Processing:", path.name)

        img = cv2.imread(str(path))
        if img is None:
            continue

        stages = enhance_scene_adaptive(img, return_stages=True)
        background = stages["whole_image_stages"]

        panels = [
            ("original", img),
            ("WB+stretch", stages["input"]),
            ("foreground", stages["foreground_output"]),
            ("bg color", background["balanced"]),
            ("bg light", background["illumination_contrast"]),
            ("final", stages["output"]),
        ]

        tiles = [label(fit(im), title) for title, im in panels]
        grid = np.hstack(tiles)

        output = Path("results/previews")
        output.mkdir(parents=True, exist_ok=True)

        save_path = output / f"{path.stem}_scene_adaptive.png"
        cv2.imwrite(str(save_path), grid)

        print("Saved:", save_path)

if __name__ == "__main__":
    main()
