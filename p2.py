import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from dip.scene_adaptive import (
    enhance_scene_adaptive,
    _scene_type,
    _preserve_luminance,
)


PANEL_W = 240
PANEL_H = 180
COLS = 8


def to_uint8(image):
    """Convert normalized float image to displayable BGR uint8."""
    return np.clip(image * 255.0, 0, 255).round().astype(np.uint8)


def wb_stretch_stages(source):
    """
    Reproduce _global_color_precondition() step by step, exposing
    intermediate images without changing the enhancement algorithm.
    """
    scene = _scene_type(source)

    normalized = source.astype(np.float32) / 255.0
    blue, green, red = cv2.split(normalized)
    means = np.mean(normalized, axis=(0, 1))

    # 1. Red compensation
    red_delta = means[1] - means[2]
    red = np.clip(
        red
        + (0.6 if scene == "blue" else 1.0)
        * red_delta
        * (1.0 - red)
        * green,
        0.0,
        1.0,
    )
    red_comp = cv2.merge([blue, green, red])

    # 2. Conditional blue compensation
    blue_applied = means[0] < 0.7 * means[1]
    if blue_applied:
        blue = np.clip(
            blue + (means[1] - means[0]) * (1.0 - blue) * green,
            0.0,
            1.0,
        )
    blue_comp = cv2.merge([blue, green, red])

    # 3. Shades-of-Gray channel balancing
    corrected = blue_comp
    p = 6.0
    illumination = np.power(
        np.mean(np.power(corrected, p), axis=(0, 1)),
        1.0 / p,
    )
    gains = np.clip(
        np.mean(illumination) / (illumination + 1e-6),
        0.6,
        2.0,
    )
    balanced_float = np.clip(corrected * gains, 0.0, 1.0)
    balanced = to_uint8(balanced_float)

    # 4. Luminance preservation (uses original source as reference)
    luminance_preserved = _preserve_luminance(balanced, source)

    # 5a. Haze offset subtraction
    values = luminance_preserved.astype(np.float32)
    low = np.minimum(
        np.percentile(values.reshape(-1, 3), 1.0, axis=0),
        60.0,
    )
    offset_values = np.clip(values - low, 0.0, None)
    haze_offset = np.clip(offset_values, 0.0, 255.0).astype(np.uint8)

    # 5b. Contrast stretch + small brightness offset
    high = float(np.percentile(offset_values.max(axis=2), 99.5))
    stretched_values = offset_values / max(high, 1.0) * 239.0 + 6.0
    haze_stretched = np.clip(stretched_values, 0.0, 255.0).astype(np.uint8)

    # 6. Lightness normalization (part of WB+stretch preview group)
    lab = cv2.cvtColor(haze_stretched, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32)
    mean = float(np.mean(lightness))
    if mean < 1.0 or abs(mean - 132.0) < 1.0:
        lightness_normalized = haze_stretched.copy()
    else:
        gamma = np.clip(
            np.log(132.0 / 255.0) / np.log(mean / 255.0),
            0.82,
            1.18,
        )
        adjusted = np.power(lightness / 255.0, gamma) * 255.0
        lab[..., 0] = np.clip(adjusted, 0.0, 255.0).round().astype(np.uint8)
        lightness_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return {
        "scene": scene,
        "blue_applied": blue_applied,
        "red_comp": to_uint8(red_comp),
        "blue_comp": to_uint8(blue_comp),
        "balanced": balanced,
        "luminance_preserved": luminance_preserved,
        "haze_offset": haze_offset,
        "haze_stretched": haze_stretched,
        "lightness_normalized": lightness_normalized,
        "gains": gains,
        "low": low,
        "high": high,
    }


def fit_panel(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    scale = min(PANEL_W / w, PANEL_H / h)
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    y = (PANEL_H - resized.shape[0]) // 2
    x = (PANEL_W - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def add_label(image, title):
    panel = fit_panel(image)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 25), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title[:30],
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_grid(panels):
    tiles = [add_label(image, title) for title, image in panels]
    rows = []
    for start in range(0, len(tiles), COLS):
        row_tiles = tiles[start:start + COLS]
        while len(row_tiles) < COLS:
            row_tiles.append(
                np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            )
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def process_one(path, output_dir):
    source = cv2.imread(str(path))
    if source is None:
        print(f"Skipping unreadable image: {path}")
        return

    wb = wb_stretch_stages(source)
    stages = enhance_scene_adaptive(source, return_stages=True)
    whole = stages["whole_image_stages"]

    mask = (stages["foreground_weight"] * 255).clip(0, 255).astype(np.uint8)

    panels = [
        ("Original", source),
        ("Red compensation", wb["red_comp"]),
        (
            "Blue compensation" if wb["blue_applied"] else "Blue comp: skipped",
            wb["blue_comp"],
        ),
        ("Channel balancing", wb["balanced"]),
        ("Luminance preserve", wb["luminance_preserved"]),
        ("Haze offset", wb["haze_offset"]),
        ("Haze stretch", wb["haze_stretched"]),
        ("Lightness normalize", wb["lightness_normalized"]),
        ("Foreground mask", mask),
        ("Foreground context", stages["denoised_input"]),
        ("Dehazed candidate", stages["dehazed"]),
        (
            "Dehaze blend",
            cv2.addWeighted(
                stages["denoised_input"],
                1.0 - stages["dehaze_amount"],
                stages["dehazed"],
                stages["dehaze_amount"],
                0,
            ),
        ),
        ("Object dehaze", stages["object_dehaze"]),
        ("Color recovery", stages["balanced"]),
        ("Red restored", stages["red_restored"]),
        ("Denoised", stages["denoised"]),
        ("CLAHE / contrast", stages["illumination_contrast"]),
        ("Foreground composite", stages["foreground_output"]),
    ]

    if whole is not None:
        panels.extend([
            ("Background dehazed", whole["dehazed"]),
            ("Background color", whole["balanced"]),
            ("Background contrast", whole["illumination_contrast"]),
        ])

    panels.append(("Final output", stages["output"]))

    grid = make_grid(panels)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{path.stem}_all_stages.png"
    cv2.imwrite(str(save_path), grid)

    print(f"Saved: {save_path}")
    print(f"Scene: {wb['scene']}")
    print(f"Blue compensation applied: {wb['blue_applied']}")
    print(f"Shades-of-Gray gains (B,G,R): {wb['gains']}")
    print(f"Haze low offsets (B,G,R): {wb['low']}")
    print(f"Haze stretch high percentile: {wb['high']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview every WB+Stretch and Scene Adaptive stage."
    )
    parser.add_argument("--n", type=int, default=10,
                        help="Number of random images (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for repeatable selection")
    parser.add_argument("--input", type=str, default="data/raw-890",
                        help="Image file or folder (default: data/raw-890)")
    parser.add_argument("--output", type=str, default="results/previews_all",
                        help="Output folder")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        selected = [input_path]
    elif input_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
        if not images:
            raise SystemExit(f"No images found in {input_path}")

        if args.seed is not None:
            random.seed(args.seed)
        selected = random.sample(images, min(args.n, len(images)))
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")

    output_dir = Path(args.output)
    for path in selected:
        print(f"\nProcessing: {path.name}")
        process_one(path, output_dir)


if __name__ == "__main__":
    main()
