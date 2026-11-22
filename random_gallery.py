"""Random gallery: images drawn by a fixed random seed, NOT chosen by score.

Layout matches fig_failures.png: one row per image, columns
  Raw | Reference | C | red comp alpha 0.8 + Shades of Gray   (last column optional)
Under each output panel: the balanced score against the reference (the old 0-1 score,
the same kind of number shown in fig_failures.png).

Images are sampled with random.Random(seed) from the sorted list of the chosen split.
Whatever comes out is shown, including bad ones. State the seed in the caption.

Put next to evaluate_all.py / candidate_pipelines.py / red_comp_study.py and run:
    python random_gallery.py --configs results\\candidate_pipelines\\selected_configs.json
Options: --n 8  --seed 42  --split test|unused  --no-two-stage
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
from skimage.color import rgb2lab

from candidate_pipelines import metrics_of, pipe_c
from dip.color_correction import correct
from evaluate_all import RAW_DEFAULT, REF_DEFAULT, pair_files, split_pairs
from red_comp_study import red_variant

OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "random_gallery"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--reference", type=Path, default=REF_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n", type=int, default=8, help="Number of random images (default 8).")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42).")
    ap.add_argument("--split", choices=["test", "unused"], default="test",
                    help="test = the 267 test images (default, same population as the "
                         "failure figure); unused = tuning images not used for tuning.")
    ap.add_argument("--tune-n", type=int, default=150)
    ap.add_argument("--data-seed", type=int, default=42, help="Seed of the train/test split.")
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--no-two-stage", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = json.loads(args.configs.read_text(encoding="utf-8"))["winners"]["C_red_sog_clahe"]["params"]
    pairs, _ = pair_files(args.input, args.reference)
    tune_pairs, test_pairs = split_pairs(pairs, args.train_fraction, args.data_seed)
    pool = sorted(test_pairs if args.split == "test" else tune_pairs[args.tune_n:],
                  key=lambda pr: pr[0].name)
    chosen = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    print(f"Sampled {len(chosen)} of {len(pool)} images ({args.split} split, seed {args.seed}):")

    cols = ["Raw", "Reference", "C"] + ([] if args.no_two_stage else ["Red 0.8 + SoG"])
    fig, axes = plt.subplots(len(chosen), len(cols), figsize=(3.6 * len(cols), 2.5 * len(chosen)),
                             squeeze=False)
    rows = []
    for r, (src, refp) in enumerate(chosen):
        img, ref = cv2.imread(str(src)), cv2.imread(str(refp))
        ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        outs = {"Raw": img, "Reference": ref, "C": pipe_c(img, params)}
        if not args.no_two_stage:
            outs["Red 0.8 + SoG"] = correct(red_variant(img, 0.8, "full"), "shades_of_gray")
        row = {"image": src.name}
        for c, name in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(cv2.cvtColor(outs[name], cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=11)
            if c == 0:
                ax.set_ylabel(src.name, fontsize=8)
            if name != "Reference":
                m = metrics_of(outs[name], ref, ref_lab, with_uciqe=False)
                row[f"{name}_balanced"] = round(m["balanced"], 3)
                ax.set_xlabel(f"score {m['balanced']:.3f}", fontsize=9)
        print(f"  {src.name}  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "image"))
        rows.append(row)
    fig.tight_layout()
    out_png = args.output / "fig_random_gallery.png"
    fig.savefig(out_png, dpi=150)
    with open(args.output / "random_gallery_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Done. Figure: {out_png}")


if __name__ == "__main__":
    main()
