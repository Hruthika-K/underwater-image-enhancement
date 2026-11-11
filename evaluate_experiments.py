"""Run controlled enhancement ablations on paired full-resolution images."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from dip.fusion import fuse
from metrics import evaluate


ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw-890"
REFERENCE_DIR = ROOT / "data" / "reference-890"
RESULTS_DIR = ROOT / "results" / "evaluations"
DEFAULT_IMAGES = (
    "101_img_.png",
    "105_img_.png",
    "106_img_.png",
    "107_img_.png",
    "10945.png",
)


EXPERIMENTS = {
    "C1_current_clahe": {},
    "C2_retinex": {},
    "C3_adaptive_gamma": {},
    "C4_multiscale_alternative_fusion": {},
}


def _lab_lightness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab, lab[..., 0].astype(np.float32)


def _from_lab_lightness(lab, lightness):
    result = lab.copy()
    result[..., 0] = np.clip(lightness, 0.0, 255.0).round().astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def _retinex(image):
    lab, lightness = _lab_lightness(image)
    value = lightness / 255.0 + 1e-3
    log_value = np.log(value)
    retinex = np.zeros_like(log_value)
    for sigma in (15.0, 80.0, 250.0):
        blurred = cv2.GaussianBlur(value, (0, 0), sigma)
        retinex += log_value - np.log(blurred + 1e-3)
    retinex /= 3.0
    low, high = np.percentile(retinex, [1.0, 99.0])
    normalized = np.clip((retinex - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    enhanced = 0.65 * (lightness / 255.0) + 0.35 * normalized
    return _from_lab_lightness(lab, enhanced * 255.0)


def _adaptive_gamma(image):
    lab, lightness = _lab_lightness(image)
    normalized = lightness / 255.0
    mean = float(normalized.mean())
    low = float(np.percentile(normalized, 5))
    high = float(np.percentile(normalized, 95))
    if mean < 0.42:
        gamma = 0.82
    elif mean < 0.55:
        gamma = 0.92
    elif high - low < 0.35:
        gamma = 0.96
    else:
        gamma = 1.0
    enhanced = np.power(normalized, gamma)
    return _from_lab_lightness(lab, enhanced * 255.0)


def _local_contrast_weight(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))
    scale = max(float(np.percentile(detail, 90)), 1.0)
    return np.clip(detail / scale, 0.0, 1.0)


def _multiscale_alternative_fusion(image):
    base = fuse(
        image,
        enable_contrast=False,
        pipeline="lean",
    )
    clahe = fuse(image, pipeline="lean")
    retinex = _retinex(base)
    gamma = _adaptive_gamma(base)
    candidates = [clahe, retinex, gamma]
    weights = [_local_contrast_weight(candidate) for candidate in candidates]
    total = sum(weights) + 1e-3
    output = sum(
        candidate.astype(np.float32) * weight[..., None]
        for candidate, weight in zip(candidates, weights)
    ) / total[..., None]
    return np.clip(output, 0.0, 255.0).round().astype(np.uint8)


def _enhance(image, configuration):
    if configuration == "C1_current_clahe":
        return fuse(image)
    base = fuse(image, enable_contrast=False, pipeline="lean")
    if configuration == "C2_retinex":
        return _retinex(base)
    if configuration == "C3_adaptive_gamma":
        return _adaptive_gamma(base)
    if configuration == "C4_multiscale_alternative_fusion":
        return _multiscale_alternative_fusion(image)
    raise ValueError(f"Unknown configuration: {configuration}")


def _read_pair(name):
    image = cv2.imread(str(RAW_DIR / name))
    reference = cv2.imread(str(REFERENCE_DIR / name))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {RAW_DIR / name}")
    if reference is None:
        raise FileNotFoundError(
            f"Could not read reference image: {REFERENCE_DIR / name}"
        )
    if image.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for {name}: {image.shape} vs {reference.shape}"
        )
    return image, reference


def run(image_names=DEFAULT_IMAGES):
    rows = []
    for name in image_names:
        image, reference = _read_pair(name)
        for experiment in EXPERIMENTS:
            output = _enhance(image, experiment)
            if output.dtype != np.uint8 or output.shape != image.shape:
                raise ValueError(f"Invalid output for {experiment}/{name}")
            scores = evaluate(output, reference)
            rows.append({
                "image": name,
                "experiment": experiment,
                **scores,
            })
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["experiment"], []).append(row)
    summary = []
    for experiment, experiment_rows in grouped.items():
        metrics = ["UIQM", "UCIQE", "PSNR", "SSIM"]
        summary.append({
            "experiment": experiment,
            **{
                metric: float(np.mean([row[metric] for row in experiment_rows]))
                for metric in metrics
            },
        })
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run()
    summary = summarize(rows)
    with (RESULTS_DIR / "controlled_ablation_rows.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "controlled_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    for row in summary:
        print(
            row["experiment"],
            "UIQM={UIQM:.4f} UCIQE={UCIQE:.4f} "
            "PSNR={PSNR:.4f} SSIM={SSIM:.4f}".format(**row),
        )


if __name__ == "__main__":
    main()
