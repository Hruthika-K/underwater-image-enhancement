"""Reproducible paired evaluation of the project's underwater enhancement methods.

Example:
    python evaluate.py --input data/raw-890 --reference data/reference-890

The default fixed pipeline matches preview.py. Full-reference metrics are
computed only for same-size pairs; images are never silently resized.
"""
import argparse
import csv
import random
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from adaptive_pipeline import enhance_adaptive
from dip.color_correction import correct
from dip.contrast import enhance_contrast
from dip.dehazing import dehaze
from metrics import evaluate as evaluate_metrics, uiqm, uciqe


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RAW_DEFAULT = ROOT / "data" / "raw-890"
REF_DEFAULT = ROOT / "data" / "reference-890"
OUT_DEFAULT = ROOT / "results" / "evaluation"


def _load(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _paired_reference(source, reference_dir):
    if not reference_dir.is_dir():
        return None
    matches = [p for p in reference_dir.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
               and p.stem.casefold() == source.stem.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous reference stem '{source.stem}': {matches}")
    return matches[0] if matches else None


def _variants(image):
    """Return named methods/ablations using the preview.py fixed settings."""
    elapsed = {}
    def timed(name, operation):
        start = time.perf_counter()
        result = operation()
        elapsed[name] = time.perf_counter() - start
        return result

    elapsed["original"] = 0.0
    udcp_only = timed("udcp_only", lambda: dehaze(
        image, method="udcp", omega=0.55, t0=0.50))
    clahe_only = timed("clahe_only", lambda: enhance_contrast(
        image, method="clahe", clip_limit=1.2))
    red = timed("red_only", lambda: correct(
        image, "red_comp", alpha=0.40, reference="blue_green"))
    gray = timed("shades_stage", lambda: correct(red, "shades_of_gray"))
    udcp = timed("udcp_stage", lambda: dehaze(
        gray, method="udcp", omega=0.55, t0=0.50))
    fixed = timed("clahe_stage", lambda: enhance_contrast(
        udcp, method="clahe", clip_limit=1.2))

    sog_only = timed("shades_of_gray_only", lambda: correct(
        image, "shades_of_gray"))
    color_clahe = timed("color_clahe_stage", lambda: enhance_contrast(
        gray, method="clahe", clip_limit=1.2))
    start = time.perf_counter()
    adaptive = enhance_adaptive(image)
    elapsed["adaptive_full_pipeline"] = time.perf_counter() - start
    methods = {
        "original": image,
        "udcp_only": udcp_only,
        "clahe_only": clahe_only,
        "red_only": red,
        "shades_of_gray_only": sog_only,
        "red_plus_shades_of_gray": gray,
        "color_correction_plus_udcp": udcp,
        "color_correction_plus_clahe": color_clahe,
        "fixed_full_pipeline": fixed,
        "adaptive_full_pipeline": adaptive,
    }
    runtimes = {
        "original": 0.0,
        "udcp_only": elapsed["udcp_only"],
        "clahe_only": elapsed["clahe_only"],
        "red_only": elapsed["red_only"],
        "shades_of_gray_only": elapsed["shades_of_gray_only"],
        "red_plus_shades_of_gray": elapsed["red_only"] + elapsed["shades_stage"],
        "color_correction_plus_udcp": elapsed["red_only"] + elapsed["shades_stage"] + elapsed["udcp_stage"],
        "color_correction_plus_clahe": elapsed["red_only"] + elapsed["shades_stage"] + elapsed["color_clahe_stage"],
        "fixed_full_pipeline": elapsed["red_only"] + elapsed["shades_stage"] + elapsed["udcp_stage"] + elapsed["clahe_stage"],
        "adaptive_full_pipeline": elapsed["adaptive_full_pipeline"],
    }
    return methods, runtimes


def _full_reference_scores(image, reference):
    if reference is None:
        return {"PSNR": "", "SSIM": "", "CIEDE2000": "",
                "reference_status": "missing"}
    if image.shape != reference.shape:
        return {"PSNR": "", "SSIM": "", "CIEDE2000": "",
                "reference_status": "shape_mismatch"}
    scores = evaluate_metrics(image, reference)
    image_lab = rgb2lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ref_lab = rgb2lab(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    scores["CIEDE2000"] = float(np.mean(deltaE_ciede2000(image_lab, ref_lab)))
    # Matching filenames and dimensions do not prove geometric alignment.
    scores["reference_status"] = "same_shape_alignment_unverified"
    return scores


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows):
    metrics = ("runtime_seconds", "UIQM", "UCIQE", "PSNR", "SSIM", "CIEDE2000")
    output = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        summary = {"method": method, "images": len(method_rows)}
        for metric in metrics:
            values = [float(r[metric]) for r in method_rows
                      if r.get(metric, "") not in ("", None)]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else ""
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--reference", type=Path, default=REF_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional deterministic limit; images are sorted by filename.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional reproducible random sample size from the sorted image list.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used with --sample (default: 42).")
    parser.add_argument("--save-images", action="store_true",
                        help="Save each method's output under output/images/<method>/.")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be positive")
    if args.limit is not None and args.sample is not None:
        parser.error("Choose either --limit or --sample, not both")
    sources = sorted((p for p in args.input.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                     key=lambda p: p.name.casefold())
    if args.limit is not None:
        sources = sources[:args.limit]
    if args.sample is not None:
        sources = sorted(random.Random(args.seed).sample(sources, min(args.sample, len(sources))),
                         key=lambda p: p.name.casefold())
    if not sources:
        parser.error(f"No supported images found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "input_directory": str(args.input.resolve()),
        "reference_directory": str(args.reference.resolve()),
        "selected_images": len(sources),
        "sampling": "random_sample_sorted_output" if args.sample is not None else "sorted_all_or_limit",
        "sample_size": args.sample,
        "seed": args.seed if args.sample is not None else None,
        "fixed_baseline": {"source": "preview.py", "red_alpha": 0.40,
                           "red_reference": "blue_green", "udcp_omega": 0.55,
                           "udcp_t0": 0.50, "clahe_clip_limit": 1.2},
        "full_reference_alignment": "same filename stem and shape checked; spatial alignment not verified",
    }
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, manifest = [], []
    for index, source in enumerate(sources, start=1):
        original = _load(source)
        if original is None:
            manifest.append({"image": source.name, "status": "unreadable"})
            continue
        reference_path = _paired_reference(source, args.reference)
        reference = _load(reference_path) if reference_path else None
        if reference_path and reference is None:
            manifest.append({"image": source.name, "status": "unreadable_reference",
                             "reference": reference_path.name})
            reference_path, reference = None, None

        variants, runtimes = _variants(original)
        for method, output in variants.items():
            score = _full_reference_scores(output, reference)
            row = {"image": source.name,
                   "reference": reference_path.name if reference_path else "",
                   "method": method, "runtime_seconds": runtimes[method],
                   "UIQM": "", "UCIQE": "", **score}
            try:
                row["UIQM"] = uiqm(output)
                row["UCIQE"] = uciqe(output)
            except Exception as exc:
                row["metric_error"] = str(exc)
            rows.append(row)
            if args.save_images:
                folder = args.output / "images" / method
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{source.stem}.png"), output)
        manifest.append({"image": source.name, "reference": reference_path.name if reference_path else None,
                         "reference_status": "same_shape_alignment_unverified" if reference is not None and reference.shape == original.shape
                         else ("shape_mismatch" if reference is not None else "missing")})
        if index % 10 == 0 or index == len(sources):
            print(f"Processed {index}/{len(sources)} images", flush=True)

    detail_fields = ["image", "reference", "method", "runtime_seconds", "UIQM", "UCIQE",
                     "PSNR", "SSIM", "CIEDE2000", "reference_status", "metric_error"]
    summary_rows = _summary(rows)
    summary_fields = list(summary_rows[0]) if summary_rows else ["method", "images"]
    _write_csv(args.output / "per_image.csv", rows, detail_fields)
    _write_csv(args.output / "summary.csv", summary_rows, summary_fields)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Scored {len({r['image'] for r in rows})} images across "
          f"{len({r['method'] for r in rows})} variants.")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
