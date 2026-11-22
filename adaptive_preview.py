
"""
Degradation-aware adaptive underwater enhancement preview.

Uses:
1. color_correction.py
2. dehazing.py
3. contrast.py

Does NOT call scene_adaptive.py, foreground masks,
background branches, or fusion.
"""

import argparse
import random
import json
from pathlib import Path

import cv2
import numpy as np

from dip.color_correction import correct
from dip.dehazing import dehaze
from dip.contrast import enhance_contrast

from adaptive_pipeline import (
    estimate_degradation,
    select_adaptive_parameters,
)

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw-890"
REF_DIR = ROOT / "data" / "reference-890"
OUT_DIR = ROOT / "results" / "previews_adaptive"

PANEL = 256

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def fit(img):

    if img is None:
        img = np.zeros(
            (PANEL, PANEL, 3),
            dtype=np.uint8
        )

    if img.ndim == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    h, w = img.shape[:2]

    scale = PANEL / max(h, w)

    resized = cv2.resize(
        img,
        (
            max(1, int(w * scale)),
            max(1, int(h * scale))
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_LINEAR
        ),
    )

    canvas = np.zeros(
        (PANEL, PANEL, 3),
        dtype=np.uint8
    )

    y = (PANEL - resized.shape[0]) // 2
    x = (PANEL - resized.shape[1]) // 2

    canvas[
        y:y + resized.shape[0],
        x:x + resized.shape[1]
    ] = resized

    return canvas


def label(img, title):

    panel = fit(img)

    cv2.rectangle(
        panel,
        (0, 0),
        (PANEL, 25),
        (0, 0, 0),
        -1
    )

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


def find_reference(raw_path, reference_dir):

    """Find paired reference image with same filename stem."""

    if not reference_dir.exists():
        return None

    for candidate in reference_dir.iterdir():

        if (
            candidate.is_file()
            and candidate.suffix.lower() in IMAGE_EXTENSIONS
        ):

            if candidate.stem.lower() == raw_path.stem.lower():
                return candidate

    return None


def process_one(image_path, reference_dir, output_dir):

    original = cv2.imread(str(image_path))

    if original is None:
        print(f"Skipping unreadable image: {image_path}")
        return None

    reference_path = find_reference(
        image_path,
        reference_dir
    )

    reference = (
        cv2.imread(str(reference_path))
        if reference_path
        else None
    )

    # --------------------------------------------------
    # STEP 1: ESTIMATE DEGRADATION
    # --------------------------------------------------

    degradation = estimate_degradation(original)

    params = select_adaptive_parameters(degradation)

    # --------------------------------------------------
    # STEP 2: ADAPTIVE RED COMPENSATION
    # --------------------------------------------------

    red_corrected = correct(
        original,
        "red_comp",
        alpha=params["red_alpha"],
        reference=params["red_reference"],
    )

    # --------------------------------------------------
    # STEP 3: SHADES OF GRAY
    # --------------------------------------------------

    color_balanced = correct(
        red_corrected,
        "shades_of_gray",
        power=6.0,
        max_gain=params["shades_max_gain"],
        low=1.0,
        high=99.0,
        stretch=params["shades_stretch"],
    )

    # --------------------------------------------------
    # STEP 4: ADAPTIVE UDCP
    # --------------------------------------------------

    dehazed = dehaze(
        color_balanced,
        method="udcp",
        patch=15,
        omega=params["udcp_omega"],
        t0=params["udcp_t0"],
        refine=True,
    )

    # --------------------------------------------------
    # STEP 5: ADAPTIVE CLAHE
    # --------------------------------------------------

    contrast_enhanced = enhance_contrast(
        dehazed,
        method="clahe",
        clip_limit=params["clahe_clip_limit"],
        tile_size=8,
    )

    # --------------------------------------------------
    # COMPARISON PANELS
    # --------------------------------------------------

    panels = [
        ("01 Original", original),

        ("02 Reference", reference),

        (
            f"03 Red alpha={params['red_alpha']:.2f}",
            red_corrected
        ),

        ("04 Shades of Gray", color_balanced),

        (
            f"05 UDCP w={params['udcp_omega']:.2f}",
            dehazed
        ),

        (
            f"06 Final c={params['clahe_clip_limit']:.2f}",
            contrast_enhanced
        ),
    ]

    tiles = [
        label(img, title)
        for title, img in panels
    ]

    grid = np.hstack(tiles)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = (
        output_dir /
        f"{image_path.stem}_adaptive_pipeline.png"
    )

    cv2.imwrite(
        str(save_path),
        grid
    )

    print(f"Saved: {save_path}")

    if reference_path:
        print(f"Reference: {reference_path.name}")
    else:
        print(
            f"No matching reference found for {image_path.name}"
        )

    print("\nDegradation indicators:")

    for key, value in degradation.items():
        print(f"  {key}: {value:.4f}")

    print("\nAdaptive parameters:")

    for key, value in params.items():
        print(f"  {key}: {value}")

    return {
        "image": str(image_path),

        "reference": (
            str(reference_path)
            if reference_path
            else None
        ),

        "comparison": str(save_path),

        "degradation": degradation,

        "parameters": params,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Adaptive underwater enhancement preview."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of random images (default: 10)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DIR),
        help="Input image file or folder"
    )

    parser.add_argument(
        "--reference",
        type=str,
        default=str(REF_DIR),
        help="Reference image folder"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUT_DIR),
        help="Output folder"
    )

    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be greater than zero")

    input_path = Path(args.input)

    if input_path.is_file():

        selected = [input_path]

    elif input_path.is_dir():

        images = [
            p for p in input_path.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]

        if not images:
            raise SystemExit(
                f"No images found in {input_path}"
            )

        if args.seed is not None:
            random.seed(args.seed)

        selected = random.sample(
            images,
            min(args.n, len(images))
        )

    else:

        raise SystemExit(
            f"Input path does not exist: {input_path}"
        )

    reference_dir = Path(args.reference)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report = []

    for image_path in selected:

        print(f"\nProcessing: {image_path.name}")

        result = process_one(
            image_path,
            reference_dir,
            output_dir
        )

        if result is not None:
            report.append(result)

    report_path = (
        output_dir / "adaptive_preview_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 45)
    print("ADAPTIVE PREVIEW COMPLETED")
    print("=" * 45)

    print(f"Processed: {len(report)} images")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
