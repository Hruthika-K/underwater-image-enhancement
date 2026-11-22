"""Candidate-pipeline comparison on the SAME split as evaluate_all.py.

Question: can a simpler, better-motivated pipeline beat Shades of Gray alone
and the fixed/adaptive pipelines on held-out UIEB pairs?

Protocol
  1. Split is rebuilt exactly as in evaluate_all.py (same seed, same fraction).
  2. Every candidate configuration is scored on (a subset of) the TUNING split.
  3. The best configuration per family is frozen.
  4. Frozen winners are evaluated ONCE on the held-out test split.

Tuning objective ("balanced", fixed in advance, no test statistics):
    mean( clip(PSNR/40), SSIM, clip(1 - CIEDE2000/30), clip(UIQM/4) )
UCIQE is reported but NOT optimised (it rewards over-saturation).

Usage:
    python candidate_pipelines.py                  # full run
    python candidate_pipelines.py --tune-n 20 --max-test 10   # quick smoke test
Put this file next to evaluate_all.py.
"""
import argparse
import itertools
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon
from skimage.color import deltaE_ciede2000, rgb2lab

from dip.color_correction import correct
from dip.contrast import enhance_contrast
from dip.dehazing import dehaze
from evaluate_all import (FIXED, RAW_DEFAULT, REF_DEFAULT, pair_files, panel,
                          split_pairs, write_csv)
from metrics import psnr, ssim, uciqe, uiqm

OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "candidate_pipelines"


# ----------------------------------------------------------------- pipelines
def _sog(img, p):
    return correct(img, "shades_of_gray", power=p.get("power", 6.0),
                   max_gain=p.get("max_gain", 2.2), stretch=p["stretch"])


def _clahe(img, clip):
    return enhance_contrast(img, method="clahe", clip_limit=clip, tile_size=8)


def pipe_a(img, p):          # A: Shades of Gray only
    return _sog(img, p)


def pipe_b(img, p):          # B: Shades of Gray -> CLAHE
    return _clahe(_sog(img, p), p["clip"])


def pipe_c(img, p):          # C: red comp -> Shades of Gray -> CLAHE
    red = correct(img, "red_comp", alpha=p["alpha"], reference="green")
    return _clahe(_sog(red, p), p["clip"])


def pipe_d(img, p):          # D: C with a weak, blended UDCP step before CLAHE
    red = correct(img, "red_comp", alpha=p["alpha"], reference="green")
    balanced = _sog(red, p)
    dh = dehaze(balanced, method="udcp", omega=p["omega"], t0=0.5)
    mixed = cv2.addWeighted(balanced, 1.0 - p["blend"], dh, p["blend"], 0)
    return _clahe(mixed, p["clip"])


def pipe_e(img, p):          # E: CLAHE first, then color (tests ordering)
    red = correct(_clahe(img, p["clip"]), "red_comp", alpha=p["alpha"], reference="green")
    return _sog(red, p)


PIPES = {"A_sog": pipe_a, "B_sog_clahe": pipe_b, "C_red_sog_clahe": pipe_c,
         "D_weak_dehaze": pipe_d, "E_clahe_first": pipe_e}
SOG_DEFAULT = {"stretch": True, "power": 6.0, "max_gain": 2.2}


def build_configs():
    cfgs = []
    for st, pw, g in itertools.product((True, False), (4.0, 6.0), (1.5, 2.2)):
        cfgs.append(("A_sog", {"stretch": st, "power": pw, "max_gain": g}))
    for st, clip in itertools.product((True, False), (1.0, 1.5, 2.5)):
        cfgs.append(("B_sog_clahe", {"stretch": st, "clip": clip}))
    for al, st, clip in itertools.product((0.2, 0.4), (True, False), (1.0, 2.0)):
        cfgs.append(("C_red_sog_clahe", {"alpha": al, "stretch": st, "clip": clip}))
    for om, bl in itertools.product((0.2, 0.35), (0.3, 0.5)):
        cfgs.append(("D_weak_dehaze", {"alpha": 0.2, "stretch": True, "clip": 1.5,
                                       "omega": om, "blend": bl}))
    for st, clip in itertools.product((True, False), (1.0, 2.0)):
        cfgs.append(("E_clahe_first", {"alpha": 0.2, "stretch": st, "clip": clip}))
    return cfgs


def fixed_pipeline(img):
    red = correct(img, "red_comp", alpha=FIXED["red_alpha"], reference=FIXED["red_reference"])
    gray = correct(red, "shades_of_gray")
    dh = dehaze(gray, method="udcp", omega=FIXED["udcp_omega"], t0=FIXED["udcp_t0"])
    return _clahe(dh, FIXED["clahe_clip_limit"])


# ------------------------------------------------------------------- metrics
def balanced(m):
    return float(np.mean([np.clip(m["PSNR"] / 40.0, 0, 1), np.clip(m["SSIM"], 0, 1),
                          np.clip(1.0 - m["CIEDE2000"] / 30.0, 0, 1),
                          np.clip(m["UIQM"] / 4.0, 0, 1)]))


def metrics_of(img, ref, ref_lab, with_uciqe=True):
    lab = rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    m = {"PSNR": psnr(img, ref), "SSIM": ssim(img, ref),
         "CIEDE2000": float(np.mean(deltaE_ciede2000(lab, ref_lab))), "UIQM": uiqm(img)}
    if with_uciqe:
        m["UCIQE"] = uciqe(img)
    m["balanced"] = balanced(m)
    m["composite"] = 0.5 * (float(np.clip(m["PSNR"] / 40.0, 0, 1)) + float(np.clip(m["SSIM"], 0, 1)))
    return m


def cfg_name(family, p):
    return family + "|" + ",".join(f"{k}={v}" for k, v in p.items())


# -------------------------------------------------------------------- tuning
def tune(tune_pairs, configs):
    sums = [{} for _ in configs]
    for i, (src, refp) in enumerate(tune_pairs, 1):
        img, ref = cv2.imread(str(src)), cv2.imread(str(refp))
        ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        for ci, (family, p) in enumerate(configs):
            m = metrics_of(PIPES[family](img, p), ref, ref_lab, with_uciqe=False)
            for k, v in m.items():
                sums[ci][k] = sums[ci].get(k, 0.0) + v
        if i % 10 == 0 or i == len(tune_pairs):
            print(f"Tune {i}/{len(tune_pairs)}", flush=True)
    rows = []
    for (family, p), s in zip(configs, sums):
        rows.append({"family": family, "config": cfg_name(family, p),
                     **{f"{k}_mean": v / len(tune_pairs) for k, v in s.items()}})
    best = {}
    for (family, p), row in zip(configs, rows):
        if family not in best or row["balanced_mean"] > best[family][2]:
            best[family] = (family, p, row["balanced_mean"])
    return rows, {f: (p, score) for f, (_, p, score) in best.items()}


# ---------------------------------------------------------------- statistics
def compare(rows, method, baseline, seed, n_boot=2000):
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["image"]] = r
    common = sorted(set(by[method]) & set(by[baseline]))
    rng = np.random.default_rng(seed)
    out = []
    for metric in ("balanced", "PSNR", "SSIM", "CIEDE2000", "UIQM"):
        a = np.array([by[method][n][metric] for n in common], dtype=float)
        b = np.array([by[baseline][n][metric] for n in common], dtype=float)
        d = (b - a) if metric == "CIEDE2000" else (a - b)      # positive = method better
        idx = rng.integers(0, len(d), (n_boot, len(d)))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        try:
            p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        out.append({"method": method, "baseline": baseline, "metric": metric,
                    "n": len(d), "mean_improvement": float(d.mean()),
                    "ci95_low": float(lo), "ci95_high": float(hi),
                    "win_rate_percent": 100.0 * float(np.mean(d > 0)),
                    "wilcoxon_p": p})
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--reference", type=Path, default=REF_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--tune-n", type=int, default=150,
                    help="Tuning images used (taken from the tuning split). Default 150.")
    ap.add_argument("--max-test", type=int, default=None, help="Development only.")
    ap.add_argument("--grid-n", type=int, default=6, help="Qualitative grids to save.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pairs, _ = pair_files(args.input, args.reference)
    tune_pairs, test_pairs = split_pairs(pairs, args.train_fraction, args.seed)
    tune_pairs = tune_pairs[:args.tune_n]
    if args.max_test:
        test_pairs = test_pairs[:args.max_test]
    configs = build_configs()
    print(f"Pairs={len(pairs)}; tune_used={len(tune_pairs)}; test={len(test_pairs)}; "
          f"configs={len(configs)}", flush=True)

    tune_rows, winners = tune(tune_pairs, configs)
    write_csv(args.output / "candidate_tuning.csv", tune_rows)
    (args.output / "selected_configs.json").write_text(json.dumps(
        {"objective": "mean(PSNR/40, SSIM, 1-CIEDE2000/30, UIQM/4), tuning split only",
         "winners": {f: {"params": p, "tune_balanced": s} for f, (p, s) in winners.items()}},
        indent=2), encoding="utf-8")
    for f, (p, s) in winners.items():
        print(f"  {f}: tune balanced={s:.4f}  {p}")

    methods = {"raw": lambda im: im,
               "sog_default": lambda im: pipe_a(im, SOG_DEFAULT),
               "fixed_pipeline": fixed_pipeline}
    for f, (p, _) in winners.items():
        methods[f"best_{f}"] = (lambda im, f=f, p=p: PIPES[f](im, p))

    rows = []
    grid_dir = args.output / "grids"
    grid_dir.mkdir(exist_ok=True)
    for i, (src, refp) in enumerate(test_pairs, 1):
        img, ref = cv2.imread(str(src)), cv2.imread(str(refp))
        ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        outs = {name: fn(img) for name, fn in methods.items()}
        for name, out in outs.items():
            rows.append({"image": src.name, "method": name, **metrics_of(out, ref, ref_lab)})
        if i <= args.grid_n:
            tiles = [panel(ref, "Reference")] + [panel(o, n) for n, o in outs.items()]
            cv2.imwrite(str(grid_dir / f"{src.stem}_candidates.png"), np.hstack(tiles))
        if i % 10 == 0 or i == len(test_pairs):
            print(f"Test {i}/{len(test_pairs)}", flush=True)
    write_csv(args.output / "candidate_test_per_image.csv", rows)

    summary = []
    for name in methods:
        sub = [r for r in rows if r["method"] == name]
        row = {"method": name, "n_images": len(sub)}
        for k in ("PSNR", "SSIM", "CIEDE2000", "UIQM", "UCIQE", "balanced", "composite"):
            vals = [r[k] for r in sub]
            row[f"{k}_mean"] = statistics.mean(vals)
            row[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(row)
    summary.sort(key=lambda r: r["balanced_mean"], reverse=True)
    write_csv(args.output / "candidate_test_summary.csv", summary)

    comps = []
    for name in methods:
        if name.startswith("best_"):
            comps += compare(rows, name, "sog_default", args.seed)
            comps += compare(rows, name, "fixed_pipeline", args.seed)
    write_csv(args.output / "candidate_comparisons.csv", comps)
    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()
