"""Make the figures for the paper: dataset pairs, qualitative comparison, failure cases.

Put this file anywhere inside your project (the project root is fine), then run it
(after the final test has finished):
    python make_figures.py

Outputs, saved in results/figures:
    fig_dataset.png        4 raw / reference pairs (random, seed 2026)
    fig_qualitative.png    one typical test image per condition group, 6 columns
    fig_failures.png       the failure-case images named in the paper: raw, reference, C
    figure_info.csv        which image went into which figure, with scores

Nothing is tuned and no results change: this only draws images. The qualitative
images are chosen by a fixed rule (the image whose gain of C over raw is the MEDIAN
of its group), not picked by eye. The scores printed under each image are the
balanced score used in the paper.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data" / "raw-890").exists())  # finds the project root from any folder
sys.path.insert(0, str(ROOT))

from candidate_pipelines import metrics_of, pipe_c
from dip.color_correction import correct
from dip.contrast import enhance_contrast
from dip.dehazing import dehaze

RAW = ROOT / "data" / "raw-890"
REF = ROOT / "data" / "reference-890"
CONFIGS = ROOT / "results" / "candidate_pipelines_calibrated" / "selected_configs.json"
TEST_SCORES = ROOT / "results" / "final_test_267" / "per_image_scores.csv"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = ["strong_red_loss", "high_haze_proxy", "green_cast", "blue_cast"]
DEFAULT_FAILURES = ["12348.png", "374_img_.png", "577_img_.png", "255_img_.png", "5818.png"]
OMEGA, T0, CLAHE_CLIP = 0.55, 0.50, 1.2          # same fixed values as the experiments


def load_selected():
    return json.loads(CONFIGS.read_text(encoding="utf-8"))["winners"]["C_red_sog_clahe"]["params"]


def build_refs():
    return {p.stem.casefold(): p for p in REF.iterdir() if p.is_file()}


def load_pair(name, refs):
    src = RAW / name
    ref_path = refs.get(src.stem.casefold())
    if not src.exists() or ref_path is None:
        raise FileNotFoundError(f"Missing raw or reference image for {name}")
    img, ref = cv2.imread(str(src)), cv2.imread(str(ref_path))
    if img is None or ref is None or img.shape != ref.shape:
        raise RuntimeError(f"Unreadable or shape-mismatched pair: {name}")
    return img, ref


def run_methods(img, selected):
    alpha = float(selected["alpha"])
    red_sog = correct(correct(img, "red_comp", alpha=alpha, reference="green"), "shades_of_gray")
    full = enhance_contrast(
        dehaze(correct(correct(img, "red_comp", alpha=0.40, reference="blue_green"),
                       "shades_of_gray"), method="udcp", omega=OMEGA, t0=T0),
        method="clahe", clip_limit=CLAHE_CLIP)
    return {
        "C": pipe_c(img, selected),
        "R + SoG": red_sog,
        "UDCP only": dehaze(img, method="udcp", omega=OMEGA, t0=T0),
        "C + UDCP": full,
    }


def score(enhanced, ref, ref_lab):
    return float(metrics_of(enhanced, ref, ref_lab, with_uciqe=False)["balanced"])


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_grid(rows, columns, path, dpi, row_labels=None, w=2.2, h=1.8):
    """rows: list of lists of (image_bgr, score_or_None). columns: header labels."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(w * len(columns), h * len(rows)),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, (img, sc) in enumerate(row):
            ax = axes[r][c]
            ax.imshow(rgb(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(columns[c], fontsize=9)
            if sc is not None:
                ax.set_xlabel(f"score {sc:.3f}", fontsize=7, labelpad=2)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("saved", path)


def read_test_scores():
    if not TEST_SCORES.exists():
        raise FileNotFoundError(f"{TEST_SCORES} not found. Run the final test first.")
    with TEST_SCORES.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_typical(rows):
    """Per group: the image whose gain of C over raw is closest to the group median."""
    by = {}
    for r in rows:
        by.setdefault(r["image"], {"groups": r["groups"].split(";")})[r["method"]] = float(r["balanced"])
    picks = {}
    for g in GROUPS:
        gains = {n: v["red_sog_clahe"] - v["raw"] for n, v in by.items() if g in v["groups"]}
        med = float(np.median(list(gains.values())))
        picks[g] = min(gains, key=lambda n: abs(gains[n] - med))
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", nargs="*", default=DEFAULT_FAILURES,
                    help="image file names for the failure figure")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    refs, selected = build_refs(), load_selected()
    info = []

    # 1. Dataset figure: 4 random raw / reference pairs
    names = sorted(p.name for p in RAW.iterdir() if p.is_file())
    sample = random.Random(2026).sample(names, 4)
    rows = []
    for n in sample:
        img, ref = load_pair(n, refs)
        rows.append([(img, None), (ref, None)])
        info.append({"figure": "dataset", "image": n, "group": "", "note": "random sample, seed 2026"})
    draw_grid(rows, ["Raw", "Reference"], OUT / "fig_dataset.png", args.dpi,
              row_labels=sample, w=3.0, h=2.2)

    # 2. Qualitative figure: typical test image per group
    picks = pick_typical(read_test_scores())
    rows, labels = [], []
    for g in GROUPS:
        n = picks[g]
        img, ref = load_pair(n, refs)
        ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        outs = run_methods(img, selected)
        row = [(img, score(img, ref, ref_lab)), (ref, None)] + \
              [(outs[k], score(outs[k], ref, ref_lab)) for k in ("C", "R + SoG", "UDCP only", "C + UDCP")]
        rows.append(row)
        labels.append(f"{g.replace('_', ' ')}\n{n}")
        info.append({"figure": "qualitative", "image": n, "group": g,
                     "note": "median gain of C over raw in this group (test split)"})
    draw_grid(rows, ["Raw", "Reference", "C", "R + SoG", "UDCP only", "C + UDCP"],
              OUT / "fig_qualitative.png", args.dpi, row_labels=labels)

    # 3. Failure cases
    rows = []
    for n in args.failures:
        img, ref = load_pair(n, refs)
        ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        c_out = run_methods(img, selected)["C"]
        s_raw, s_c = score(img, ref, ref_lab), score(c_out, ref, ref_lab)
        rows.append([(img, s_raw), (ref, None), (c_out, s_c)])
        info.append({"figure": "failures", "image": n, "group": "",
                     "note": f"raw {s_raw:.3f}, C {s_c:.3f}, change {s_c - s_raw:+.3f}"})
    draw_grid(rows, ["Raw", "Reference", "C"], OUT / "fig_failures.png", args.dpi,
              row_labels=list(args.failures), w=3.0, h=2.2)

    with (OUT / "figure_info.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "image", "group", "note"])
        w.writeheader()
        w.writerows(info)
    print("Done. Look at the three PNG files, then insert them in the paper and the slides.")


if __name__ == "__main__":
    main()