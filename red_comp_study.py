"""Red-compensation study: strength sweep + which formula terms matter.

Pipeline under study (nothing else changes):  red compensation -> Shades of Gray
(default settings).  CLAHE and dehazing are left out because earlier results showed
they add little or hurt.

Formula (as in dip/color_correction.py):
    r_new = r + alpha * (ref_mean - r_mean) * (1 - r) * g
Variants (each gets its OWN strength, chosen on the tuning split only):
    full     : the formula above
    no_1mr   : without the (1 - r) factor
    no_g     : without the g factor
    no_both  : without both factors
    ref_bg   : reference mean = (green + blue) / 2 instead of green

Protocol (same split as before)
  1. Score every tuning-config on the first --tune-n images of the tuning split.
  2. Choose each variant's alpha on the tuning split (primary score below).
  3. Write the claims to claims_locked.json BEFORE any test image is scored.
  4. Score on the 267 test images: full alpha sweep (descriptive curve) + winners.
  5. Paired tests / bootstrap CIs on the test split.

Primary score ("calibrated"): mean over PSNR, SSIM, CIEDE2000 (sign flipped), UIQM of
(value - raw mean) / raw SD, with raw mean and sample SD from the raw TUNING images.
The older "balanced" score is also reported. Always read the individual metrics too.

Put next to evaluate_all.py and candidate_pipelines.py, then run:
    python red_comp_study.py
    python red_comp_study.py --limit 10      # quick smoke test
Progress is cached; if it stops, run the same command again.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon
from skimage.color import rgb2lab

from candidate_pipelines import metrics_of
from dip.color_correction import correct
from evaluate_all import RAW_DEFAULT, REF_DEFAULT, pair_files, split_pairs, write_csv

OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "red_comp_study"
FULL_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
VAR_ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
VARIANTS = ["no_1mr", "no_g", "no_both", "ref_bg"]
RAWM = ["PSNR", "SSIM", "CIEDE2000", "UIQM"]
SHOW = ["calibrated", "balanced", "PSNR", "SSIM", "CIEDE2000", "UIQM"]


# ------------------------------------------------------------------- method
def red_variant(img, alpha, variant):
    f = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    ref_mean = (g.mean() + b.mean()) / 2.0 if variant == "ref_bg" else g.mean()
    term = alpha * (ref_mean - r.mean())
    if variant in ("full", "ref_bg", "no_g"):
        term = term * (1.0 - r)
    if variant in ("full", "ref_bg", "no_1mr"):
        term = term * g
    r2 = np.clip(r + term, 0.0, 1.0)
    return (cv2.merge([b, g, r2]) * 255).round().astype(np.uint8)


def apply(img, cfg):
    if cfg == "raw":
        return img
    variant, alpha = cfg.split("@")
    return correct(red_variant(img, float(alpha), variant), "shades_of_gray")


def tune_configs():
    return ["raw"] + [f"full@{a}" for a in FULL_ALPHAS] + \
           [f"{v}@{a}" for v in VARIANTS for a in VAR_ALPHAS]


# ------------------------------------------------------------------ scoring
def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in RAWM + ["balanced", "composite"]:
            r[k] = float(r[k])
    return rows


def score_images(pairs, configs, path, label):
    done = {(r["image"], r["config"]) for r in read_rows(path)}
    todo = [(s, rf) for s, rf in pairs if any((s.name, c) not in done for c in configs)]
    print(f"{label}: {len(pairs) - len(todo)} images fully cached, {len(todo)} to do", flush=True)
    new = not path.exists()
    t0 = time.time()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = None
        for i, (src, refp) in enumerate(todo, 1):
            img, ref = cv2.imread(str(src)), cv2.imread(str(refp))
            ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
            for cfg in configs:
                if (src.name, cfg) in done:
                    continue
                m = metrics_of(apply(img, cfg), ref, ref_lab, with_uciqe=False)
                row = {"image": src.name, "config": cfg,
                       **{k: m[k] for k in RAWM + ["balanced", "composite"]}}
                if w is None:
                    w = csv.DictWriter(fh, list(row))
                    if new:
                        w.writeheader()
                        new = False
                w.writerow(row)
            if i % 10 == 0 or i == len(todo):
                fh.flush()
                print(f"  {label} {i}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def anchors_from_raw(rows):
    raw = [r for r in rows if r["config"] == "raw"]
    return {k: (float(np.mean([r[k] for r in raw])), float(np.std([r[k] for r in raw], ddof=1)))
            for k in RAWM}


def add_calibrated(rows, anc):
    for r in rows:
        r["calibrated"] = float(np.mean([
            (r["PSNR"] - anc["PSNR"][0]) / anc["PSNR"][1],
            (r["SSIM"] - anc["SSIM"][0]) / anc["SSIM"][1],
            (anc["CIEDE2000"][0] - r["CIEDE2000"]) / anc["CIEDE2000"][1],
            (r["UIQM"] - anc["UIQM"][0]) / anc["UIQM"][1]]))
    return rows


def table(rows):
    t = {}
    for r in rows:
        t.setdefault(r["config"], {})[r["image"]] = r
    return t


def summarize(rows, configs):
    t, out = table(rows), []
    for c in configs:
        if c not in t:
            continue
        rs = list(t[c].values())
        row = {"config": c, "n_images": len(rs)}
        for k in SHOW:
            v = [r[k] for r in rs]
            row[f"{k}_mean"] = float(np.mean(v))
            row[f"{k}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return out


# --------------------------------------------------------------- statistics
def diff(t, a, b, metric):
    names = sorted(set(t[a]) & set(t[b]))
    x = np.array([t[a][n][metric] for n in names]) - np.array([t[b][n][metric] for n in names])
    return -x if metric == "CIEDE2000" else x          # positive = a better


def one_sided_p(x):
    try:
        return float(wilcoxon(x, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def two_sided_p(x):
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return 1.0


def compare_rows(t, a, b, rng):
    out = []
    for metric in SHOW:
        x = diff(t, a, b, metric)
        idx = rng.integers(0, len(x), (5000, len(x)))
        lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
        out.append({"method": a, "baseline": b, "metric": metric, "n": len(x),
                    "mean_improvement": float(x.mean()), "ci95_low": float(lo),
                    "ci95_high": float(hi), "win_rate_percent": 100.0 * float(np.mean(x > 0)),
                    "wilcoxon_two_sided_p": two_sided_p(x)})
    return out


def plot_curve(tune_s, test_s, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable: skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, label in ((tune_s, "tuning split"), (test_s, "test split")):
        pts = [(float(r["config"].split("@")[1]), r["calibrated_mean"],
                r["calibrated_std"] / np.sqrt(r["n_images"]))
               for r in s if r["config"].startswith("full@")]
        pts.sort()
        a, m, se = zip(*pts)
        ax.errorbar(a, m, yerr=[1.96 * e for e in se], marker="o", capsize=3, label=label)
    ax.set_xlabel("red-compensation strength alpha (alpha = 0: no compensation)")
    ax.set_ylabel("calibrated score (SD of raw images)")
    ax.set_title("Red compensation + Shades of Gray: strength sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--reference", type=Path, default=REF_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--tune-n", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="Development only.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pairs, _ = pair_files(args.input, args.reference)
    tune_pairs, test_pairs = split_pairs(pairs, args.train_fraction, args.seed)
    tune_pairs = tune_pairs[:args.tune_n]
    if args.limit:
        tune_pairs, test_pairs = tune_pairs[:args.limit], test_pairs[:max(3, args.limit // 2)]
    print(f"tune_used={len(tune_pairs)}  test={len(test_pairs)}", flush=True)

    # sanity: our 'full' variant must reproduce the project's red_comp function
    probe = cv2.imread(str(tune_pairs[0][0]))
    ref_fn = correct(probe, "red_comp", alpha=0.4, reference="green")
    mine = red_variant(probe, 0.4, "full")
    assert int(np.abs(ref_fn.astype(int) - mine.astype(int)).max()) <= 1, \
        "variant 'full' does not match dip.color_correction.red_comp"

    # 1. tuning
    tune_path = args.output / "scores_tune.csv"
    score_images(tune_pairs, tune_configs(), tune_path, "tune")
    tune_rows = read_rows(tune_path)
    anc = anchors_from_raw(tune_rows)
    add_calibrated(tune_rows, anc)
    tune_summary = summarize(tune_rows, tune_configs())
    write_csv(args.output / "tuning_summary.csv", tune_summary)

    best = {}
    for fam, alphas in [("full", FULL_ALPHAS)] + [(v, VAR_ALPHAS) for v in VARIANTS]:
        cands = [r for r in tune_summary if r["config"] in {f"{fam}@{a}" for a in alphas}]
        best[fam] = max(cands, key=lambda r: r["calibrated_mean"])["config"]
    print("Chosen on tuning split:", best, flush=True)

    # 2. lock claims BEFORE test scoring
    lock = args.output / "claims_locked.json"
    claims = [
        ("Tuned full red compensation beats strength alpha = 1.0", best["full"], "full@1.0"),
        ("Tuned full red compensation beats no compensation (alpha = 0)", best["full"], "full@0.0"),
        ("Full formula beats the version without the (1 - red) factor", best["full"], best["no_1mr"]),
        ("Full formula beats the version without the green factor", best["full"], best["no_g"]),
    ]
    if not lock.exists():
        lock.write_text(json.dumps({
            "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test": "one-sided paired Wilcoxon, better > worse, on the calibrated score",
            "chosen_on_tuning": best, "anchors_raw_tuning_mean_sd": anc,
            "claims": [{"claim": c, "better": a, "worse": b} for c, a, b in claims]},
            indent=2), encoding="utf-8")
        print(f"Claims locked in {lock}", flush=True)
    claims = [(c["claim"], c["better"], c["worse"])
              for c in json.loads(lock.read_text(encoding="utf-8"))["claims"]]
    chosen = json.loads(lock.read_text(encoding="utf-8"))["chosen_on_tuning"]

    # 3. test
    test_cfgs = ["raw"] + [f"full@{a}" for a in FULL_ALPHAS] + \
                [chosen[v] for v in VARIANTS]
    test_cfgs = list(dict.fromkeys(test_cfgs))
    test_path = args.output / "scores_test.csv"
    score_images(test_pairs, test_cfgs, test_path, "test")
    test_rows = add_calibrated(read_rows(test_path), anc)
    t = table(test_rows)
    test_summary = summarize(test_rows, test_cfgs)
    write_csv(args.output / "test_summary.csv", test_summary)
    write_csv(args.output / "test_per_image.csv", test_rows)

    rng = np.random.default_rng(args.seed)
    comps = []
    for _, a, b in claims:
        comps += compare_rows(t, a, b, rng)
    for v in ("no_both", "ref_bg"):
        comps += compare_rows(t, chosen["full"], chosen[v], rng)
    write_csv(args.output / "test_comparisons.csv", comps)

    claim_rows = []
    for name, a, b in claims:
        x = diff(t, a, b, "calibrated")
        claim_rows.append({"claim": name, "better": a, "worse": b, "n_images": len(x),
                           "mean_score_difference": round(float(x.mean()), 4),
                           "better_wins_percent": round(100 * float(np.mean(x > 0)), 1),
                           "wilcoxon_one_sided_p": one_sided_p(x),
                           "supported_at_0.05": one_sided_p(x) < 0.05})
    write_csv(args.output / "claims_results.csv", claim_rows)
    plot_curve(tune_summary, test_summary, args.output / "alpha_curve.png")

    print("\nTEST SUMMARY (calibrated mean | PSNR | SSIM | CIEDE2000 | UIQM)")
    for r in test_summary:
        print(f"  {r['config']:12s} {r['calibrated_mean']:+.3f} | {r['PSNR_mean']:.2f} | "
              f"{r['SSIM_mean']:.3f} | {r['CIEDE2000_mean']:.2f} | {r['UIQM_mean']:.3f}")
    print("\nCLAIMS")
    for c in claim_rows:
        print(f"  {c['claim']}: diff {c['mean_score_difference']}, p = "
              f"{c['wilcoxon_one_sided_p']:.2g}, supported = {c['supported_at_0.05']}")
    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()
