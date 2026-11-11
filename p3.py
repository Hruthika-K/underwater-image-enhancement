import argparse
import random
from pathlib import Path

import cv2

from p2 import wb_stretch_stages, make_grid
from dip.scene_adaptive import _scene_type, _enhance_whole_image


def process_one(path, output_dir):
    source = cv2.imread(str(path))
    if source is None:
        print(f"Skipping unreadable image: {path}")
        return

    # Keep WB + Stretch, but skip foreground mask and all foreground/background blending.
    wb = wb_stretch_stages(source)
    preconditioned = wb["lightness_normalized"]

    scene = _scene_type(source)
    dehaze_method = "udcp" if scene in ("blue", "mixed") else "red"

    # Apply one restrained enhancement sequence to the ENTIRE preconditioned image.
    global_stages = _enhance_whole_image(
        preconditioned,
        scene,
        dehaze_method,
        contrast_method="clahe",
        dehaze_strength=0.6,
        color_strength=0.35,
        enable_red=False,
        brightness_strength=1.8,
        contrast_strength=1.2,
    )

    panels = [
        ("Original", source),
        ("Red compensation", wb["red_comp"]),
        (
            "Blue compensation" if wb["blue_applied"] else "Blue comp: skipped",
            wb["blue_comp"],
        ),
        ("Channel balancing", wb["balanced"]),
        ("Luminance preservation", wb["luminance_preserved"]),
        ("Haze offset", wb["haze_offset"]),
        ("Haze stretch", wb["haze_stretched"]),
        ("Lightness normalization", preconditioned),
        ("Global dehazed", global_stages["dehazed"]),
        ("Global dehaze blend", global_stages["dehaze_blend"]),
        ("Global color recovery", global_stages["balanced"]),
        ("Global denoised", global_stages["denoised"]),
        ("Global CLAHE", global_stages["illumination_contrast"]),
        ("Final global output", global_stages["output"]),
    ]

    grid = make_grid(panels)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{path.stem}_global_no_mask.png"
    cv2.imwrite(str(save_path), grid)

    print(f"Saved: {save_path}")
    print(f"Scene: {scene}")
    print("Foreground mask: NOT used")
    print("Separate background branch: NOT used")
    print(f"Shades-of-Gray gains (B,G,R): {wb['gains']}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview global enhancement without foreground/background separation."
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input", default="data/raw-890")
    parser.add_argument("--output", default="results/previews_global")
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
