"""Group underwater images using simple, measurable image characteristics.

By default this reads data/raw-890 and writes CSV plus one filename list per
group under results/image_condition_groups. Groups can overlap: for example,
an image may be both blue-cast and low-contrast. Quartile thresholds are
calculated from the images in the input folder for each run.

Run from the project root:
    python group_images_by_condition.py

Or choose folders explicitly:
    python group_images_by_condition.py --input data/raw-890 --output results/my_groups

The default split is the project's 623-image tuning/train split. Use
--split all to group every image, or --split test to group the 267-image test set.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANIFEST = ROOT / "results" / "research_evaluation" / "split_manifest.json"


def measure(image):
    """Return cast, red-loss, haze-proxy, and local-contrast measurements."""
    f = image.astype(np.float32) / 255.0
    b_mean, g_mean, r_mean = f.mean(axis=(0, 1))
    red_deficiency = float(np.clip((g_mean - r_mean) / max(float(g_mean), 1e-6), 0, 1))

    if g_mean >= b_mean and g_mean > r_mean:
        cast = "green_cast"
    elif b_mean > g_mean and b_mean > r_mean:
        cast = "blue_cast"
    else:
        cast = "red_dominant_or_other"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32)
    local_mean = cv2.boxFilter(lightness, -1, (15, 15), normalize=True)
    local_square_mean = cv2.boxFilter(lightness * lightness, -1, (15, 15), normalize=True)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0))
    return {
        "cast_group": cast,
        "blue_mean": float(b_mean),
        "green_mean": float(g_mean),
        "red_mean": float(r_mean),
        "red_deficiency": red_deficiency,
        "haze_proxy": float(np.clip(1.0 - float(np.std(gray)) / 64.0, 0, 1)),
        "local_contrast": float(local_std.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw-890",
                        help="Folder containing source underwater images")
    parser.add_argument("--output", type=Path, default=None,
                        help="Folder for CSV, threshold JSON, and group filename lists")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train",
                        help="Dataset split to group (default: 623-image train/tuning split)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="Split manifest used by --split train or --split test")
    parser.add_argument("--red-loss-quantile", type=float, default=0.75,
                        help="Quantile cutoff for strong red loss (default: 0.75)")
    parser.add_argument("--haze-quantile", type=float, default=0.75,
                        help="Quantile cutoff for high haze proxy (default: 0.75)")
    parser.add_argument("--contrast-quantile", type=float, default=0.25,
                        help="Quantile cutoff for low local contrast (default: 0.25)")
    args = parser.parse_args()

    for value, label in ((args.red_loss_quantile, "red-loss"),
                         (args.haze_quantile, "haze"),
                         (args.contrast_quantile, "contrast")):
        if not 0 < value < 1:
            parser.error(f"--{label}-quantile must be between 0 and 1")
    if not args.input.is_dir():
        parser.error(f"Input folder does not exist: {args.input}")

    if args.split == "all":
        paths = sorted((p for p in args.input.iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED),
                       key=lambda p: p.name.casefold())
        split_description = "all readable images in input folder"
    else:
        if not args.manifest.is_file():
            parser.error(f"Split manifest does not exist: {args.manifest}")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        key = "tuning_images" if args.split == "train" else "test_images"
        names = manifest.get(key)
        if not names:
            parser.error(f"Manifest has no {key} list: {args.manifest}")
        paths = [args.input / name for name in names]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            parser.error(f"{len(missing)} split images are missing from {args.input}; first: {missing[0]}")
        split_description = f"{args.split} split from {args.manifest} ({len(paths)} listed images)"

    rows = []
    unreadable = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED:
            continue
        image = cv2.imread(str(path))
        if image is None:
            unreadable.append(path.name)
            continue
        rows.append({"image": path.name, **measure(image)})
    if not rows:
        parser.error(f"No readable supported images found in {args.input}")

    red_cut = float(np.quantile([r["red_deficiency"] for r in rows], args.red_loss_quantile))
    haze_cut = float(np.quantile([r["haze_proxy"] for r in rows], args.haze_quantile))
    contrast_cut = float(np.quantile([r["local_contrast"] for r in rows], args.contrast_quantile))
    groups = {name: [] for name in ("green_cast", "blue_cast", "red_dominant_or_other",
                                    "strong_red_loss", "high_haze_proxy", "low_local_contrast")}
    for row in rows:
        row["groups"] = []
        groups[row["cast_group"]].append(row["image"])
        row["groups"].append(row["cast_group"])
        if row["red_deficiency"] >= red_cut:
            groups["strong_red_loss"].append(row["image"])
            row["groups"].append("strong_red_loss")
        if row["haze_proxy"] >= haze_cut:
            groups["high_haze_proxy"].append(row["image"])
            row["groups"].append("high_haze_proxy")
        if row["local_contrast"] <= contrast_cut:
            groups["low_local_contrast"].append(row["image"])
            row["groups"].append("low_local_contrast")
        row["groups"] = ";".join(row["groups"])

    if args.output is None:
        suffix = {"train": "train623", "test": "test267", "all": "all"}[args.split]
        args.output = ROOT / "results" / f"image_condition_groups_{suffix}"
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["image", "cast_group", "blue_mean", "green_mean", "red_mean",
              "red_deficiency", "haze_proxy", "local_contrast", "groups"]
    with (args.output / "image_features_and_groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for group, names in groups.items():
        (args.output / f"{group}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    (args.output / "group_thresholds.json").write_text(json.dumps({
        "input_folder": str(args.input.resolve()),
        "split": args.split,
        "split_description": split_description,
        "split_manifest": str(args.manifest.resolve()) if args.split != "all" else None,
        "n_readable_images": len(rows),
        "n_unreadable_images": len(unreadable),
        "quantiles": {"red_loss": args.red_loss_quantile,
                      "haze_proxy": args.haze_quantile,
                      "local_contrast": args.contrast_quantile},
        "cutoffs": {"strong_red_loss_min": red_cut,
                    "high_haze_proxy_min": haze_cut,
                    "low_local_contrast_max": contrast_cut},
        "group_sizes": {name: len(names) for name, names in groups.items()},
        "unreadable_files": unreadable,
        "definitions": {
            "cast": "dominant mean BGR channel; green/blue labels require that channel to exceed red",
            "red_deficiency": "clip((mean_green - mean_red) / mean_green, 0, 1)",
            "haze_proxy": "clip(1 - grayscale standard deviation / 64, 0, 1); rough proxy, not physical haze",
            "local_contrast": "mean 15x15 local standard deviation of CIELAB lightness",
            "overlap": "color-cast groups and condition quartile groups can overlap"
        }
    }, indent=2), encoding="utf-8")
    print(f"Grouped {len(rows)} images. Results: {args.output.resolve()}")
    for group, names in groups.items():
        print(f"  {group}: {len(names)}")


if __name__ == "__main__":
    main()
