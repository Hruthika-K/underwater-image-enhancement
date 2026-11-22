"""Do the pipeline outputs contain more detail than the UIEB reference, and is some of it noise?

For each image, measure on raw, reference, pipeline C and red comp alpha 0.8 + SoG:
  grad         : mean Sobel gradient magnitude of the gray image (edge energy)
  UIQM, UCIQE  : standard no-reference underwater quality scores
  flat_hf      : CUSTOM measure (not a standard metric). Mean absolute high-pass response
                 (gray minus Gaussian blur, sigma 1.5) inside the FLAT regions of the RAW image
                 (the 30% of pixels with the lowest local std, 15x15 window). Real scene detail
                 should be low there, so a rise suggests amplified noise or compression blocks.
  flat_hf_norm : flat_hf divided by the image's overall gray std, so a plain contrast
                 increase is not counted as noise.
The flat mask comes from the raw image, so it is the same for every version.
Then reports each version minus the reference with a paired Wilcoxon test.

CAUTION: these are proxies. They cannot prove an image looks better. Look at images too.

By default it uses the tuning-split images that were never used to choose parameters
(--split unused), so the test set stays clean. Use --split test for the 267 test images.

Put next to evaluate_all.py / candidate_pipelines.py / red_comp_study.py and run:
    python detail_check.py --configs results\\candidate_pipelines\\selected_configs.json --limit 100
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon

from candidate_pipelines import pipe_c
from dip.color_correction import correct
from evaluate_all import RAW_DEFAULT, REF_DEFAULT, pair_files, split_pairs, write_csv
from metrics import uciqe, uiqm
from red_comp_study import red_variant

OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "detail_check"
KEYS = ("grad", "UIQM", "UCIQE", "flat_hf", "flat_hf_norm")


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def grad(img):
    g = gray(img)
    return float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())


def flat_mask(raw):
    g = cv2.GaussianBlur(gray(raw), (0, 0), 1.0)
    m = cv2.boxFilter(g, -1, (15, 15))
    s = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (15, 15)) - m * m, 0))
    return s <= np.percentile(s, 30)


def flat_hf(img, mask):
    g = gray(img)
    if mask.shape != g.shape:
        mask = cv2.resize(mask.astype(np.uint8), (g.shape[1], g.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5))
    val = float(hp[mask].mean())
    return val, val / (float(g.std()) + 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--reference", type=Path, default=REF_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--split", choices=["unused", "test"], default="unused",
                    help="unused = tuning-split images NOT used to choose any parameter "
                         "(default, keeps the test set clean); test = the 267 test images.")
    ap.add_argument("--tune-n", type=int, default=150,
                    help="How many leading tuning images were used for tuning (default 150).")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N images.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    p = json.loads(args.configs.read_text(encoding="utf-8"))["winners"]["C_red_sog_clahe"]["params"]
    pairs, _ = pair_files(args.input, args.reference)
    tune_pairs, test_pairs = split_pairs(pairs, args.train_fraction, args.seed)
    use = tune_pairs[args.tune_n:] if args.split == "unused" else test_pairs
    if args.limit:
        use = use[:args.limit]
    print(f"split={args.split}: {len(use)} images", flush=True)

    rows = []
    for i, (src, refp) in enumerate(use, 1):
        img, ref = cv2.imread(str(src)), cv2.imread(str(refp))
        mask = flat_mask(img)
        outs = {"raw": img, "reference": ref, "C": pipe_c(img, p),
                "red0.8_SoG": correct(red_variant(img, 0.8, "full"), "shades_of_gray")}
        for name, o in outs.items():
            fh, fn = flat_hf(o, mask)
            rows.append({"image": src.name, "version": name, "grad": grad(o),
                         "UIQM": uiqm(o), "UCIQE": uciqe(o),
                         "flat_hf": fh, "flat_hf_norm": fn})
        if i % 20 == 0 or i == len(use):
            print(f"  {i}/{len(use)}", flush=True)
    write_csv(args.output / "detail_per_image.csv", rows)

    by = {}
    for r in rows:
        by.setdefault(r["version"], {})[r["image"]] = r
    names = sorted(by["raw"])
    print("\nMEAN over images")
    for v in by:
        print(f"  {v:11s} grad {np.mean([by[v][n]['grad'] for n in names]):7.2f} | "
              f"UIQM {np.mean([by[v][n]['UIQM'] for n in names]):.3f} | "
              f"UCIQE {np.mean([by[v][n]['UCIQE'] for n in names]):.2f} | "
              f"flat_hf {np.mean([by[v][n]['flat_hf'] for n in names]):.3f} | "
              f"flat_hf_norm {np.mean([by[v][n]['flat_hf_norm'] for n in names]):.4f}")
    print("\nPAIRED: version minus reference (positive = version higher)")
    out = []
    for v in ("raw", "C", "red0.8_SoG"):
        for k in KEYS:
            d = np.array([by[v][n][k] - by["reference"][n][k] for n in names])
            try:
                pv = float(wilcoxon(d).pvalue)
            except ValueError:
                pv = 1.0
            out.append({"version": v, "metric": k, "mean_diff_vs_reference": float(d.mean()),
                        "higher_than_reference_percent": 100 * float(np.mean(d > 0)), "p": pv})
            print(f"  {v:11s} {k:12s} {d.mean():+9.4f}  higher on {100 * np.mean(d > 0):5.1f}% of images  p={pv:.2g}")
    write_csv(args.output / "detail_vs_reference.csv", out)
    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()