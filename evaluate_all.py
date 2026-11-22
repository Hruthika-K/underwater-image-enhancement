"""Split-first tuning and held-out evaluation for the classical pipeline.

Research question: Does a degradation-aware controller improve a classical
underwater enhancement pipeline over fixed parameters on held-out UIEB pairs?

Example (default: all eligible pairs, 70/30 split, seed 42):
    python evaluate_all.py

Candidate coefficients are searched on the tuning split only. Test data is
evaluated once after the selected controller is frozen. Same-name, same-shape
references are assumed paired; geometric alignment is not independently
verified, so full-reference results must be interpreted with that limitation.
"""
import argparse
import csv
import itertools
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
from skimage.color import deltaE_ciede2000, rgb2lab

from adaptive_pipeline import DEFAULT_CONTROLLER, enhance_adaptive, estimate_degradation
from dip.color_correction import correct
from dip.contrast import enhance_contrast
from dip.dehazing import dehaze
from metrics import psnr, ssim, uiqm, uciqe

ROOT = Path(__file__).resolve().parent
RAW_DEFAULT = ROOT / "data" / "raw-890"
REF_DEFAULT = ROOT / "data" / "reference-890"
OUT_DEFAULT = ROOT / "results" / "research_evaluation"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FIXED = {"red_alpha": 0.40, "red_reference": "blue_green",
         "udcp_omega": 0.55, "udcp_t0": 0.50, "clahe_clip_limit": 1.2}


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_files(raw_dir, ref_dir):
    refs = {}
    for p in ref_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            refs.setdefault(p.stem.casefold(), []).append(p)
    pairs, excluded = [], []
    for src in sorted((p for p in raw_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTS),
                      key=lambda p: p.name.casefold()):
        candidates = refs.get(src.stem.casefold(), [])
        if len(candidates) != 1:
            excluded.append({"image": src.name, "reason": "missing_or_ambiguous_reference"})
            continue
        image, ref = cv2.imread(str(src)), cv2.imread(str(candidates[0]))
        if image is None or ref is None:
            excluded.append({"image": src.name, "reason": "unreadable_pair"})
        elif image.shape != ref.shape:
            excluded.append({"image": src.name, "reference": candidates[0].name,
                             "reason": "shape_mismatch"})
        else:
            pairs.append((src, candidates[0]))
    return pairs, excluded


def split_pairs(pairs, train_fraction, seed):
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    boundary = int(len(shuffled) * train_fraction)
    return shuffled[:boundary], shuffled[boundary:]


def candidate_configs(count, seed):
    grid = list(itertools.product(
        (0.05, 0.15), (0.15, 0.25, 0.35),
        (0.45, 0.55), (0.15, 0.25, 0.35), (0.4, 0.8, 1.2)))
    rng = random.Random(seed)
    selected = rng.sample(grid, min(count, len(grid)))
    configs = [dict(zip(("red_alpha_base", "red_alpha_slope", "udcp_omega_base",
                         "udcp_omega_slope", "clahe_slope"), values))
               for values in selected]
    default = dict(DEFAULT_CONTROLLER)
    if default not in configs:
        configs.append(default)
    return configs


def tuning_score(output, reference):
    # Fixed bounded composite: equal weight to SSIM and PSNR mapped to [0, 1]
    # over a predeclared 0–40 dB range. No test-set statistics are used.
    p = float(np.clip(psnr(output, reference) / 40.0, 0.0, 1.0))
    s = float(np.clip(ssim(output, reference), 0.0, 1.0))
    return 0.5 * (p + s)


def tune(train_pairs, configs, progress_every=25):
    records = []
    for ci, config in enumerate(configs, 1):
        scores = []
        for i, (source, refpath) in enumerate(train_pairs, 1):
            image, ref = cv2.imread(str(source)), cv2.imread(str(refpath))
            result = enhance_adaptive(image, controller=config)
            scores.append(tuning_score(result, ref))
            if i % progress_every == 0 or i == len(train_pairs):
                print(f"Tune candidate {ci}/{len(configs)}: {i}/{len(train_pairs)}",
                      flush=True)
        records.append({"controller": config, "mean_composite": float(np.mean(scores)),
                        "n_images": len(scores)})
    records.sort(key=lambda r: r["mean_composite"], reverse=True)
    return records[0]["controller"], records


def run_variants(image, controller):
    """Return required methods plus one-stage-removed ablations."""
    elapsed = {}
    def measured(key, fn):
        start = time.perf_counter()
        value = fn()
        elapsed[key] = time.perf_counter() - start
        return value

    red = measured("red_only", lambda: correct(image, "red_comp", alpha=FIXED["red_alpha"],
                                                reference=FIXED["red_reference"]))
    gray = measured("shades_after_red", lambda: correct(red, "shades_of_gray"))
    gray_no_red = measured("shades_without_red", lambda: correct(image, "shades_of_gray"))
    sog_only = measured("shades_only", lambda: correct(image, "shades_of_gray"))
    udcp_only = measured("udcp_only", lambda: dehaze(image, method="udcp", omega=FIXED["udcp_omega"], t0=FIXED["udcp_t0"]))
    corr_udcp = measured("udcp_after_color", lambda: dehaze(gray, method="udcp", omega=FIXED["udcp_omega"], t0=FIXED["udcp_t0"]))
    fixed = measured("clahe_after_udcp", lambda: enhance_contrast(corr_udcp, method="clahe", clip_limit=FIXED["clahe_clip_limit"]))
    clahe_only = measured("clahe_only", lambda: enhance_contrast(image, method="clahe", clip_limit=FIXED["clahe_clip_limit"]))
    no_red_udcp = measured("udcp_without_red", lambda: dehaze(gray_no_red, method="udcp", omega=FIXED["udcp_omega"], t0=FIXED["udcp_t0"]))
    no_red = measured("clahe_without_red", lambda: enhance_contrast(no_red_udcp, method="clahe", clip_limit=FIXED["clahe_clip_limit"]))
    no_gray_udcp = measured("udcp_without_shades", lambda: dehaze(red, method="udcp", omega=FIXED["udcp_omega"], t0=FIXED["udcp_t0"]))
    no_gray = measured("clahe_without_shades", lambda: enhance_contrast(no_gray_udcp, method="clahe", clip_limit=FIXED["clahe_clip_limit"]))
    no_udcp = measured("clahe_without_udcp", lambda: enhance_contrast(gray, method="clahe", clip_limit=FIXED["clahe_clip_limit"]))
    no_clahe = corr_udcp
    elapsed["raw"] = 0.0
    adaptive_start = time.perf_counter()
    adaptive, details = enhance_adaptive(image, return_details=True, controller=controller)
    elapsed["adaptive"] = time.perf_counter() - adaptive_start
    methods = {
        "raw": image,
        "red_comp_only": red,
        "shades_of_gray_only": sog_only,
        "udcp_only": udcp_only,
        "clahe_only": clahe_only,
        "fixed_pipeline": fixed,
        "adaptive_pipeline": adaptive,
        "ablation_no_red_comp": no_red,
        "ablation_no_shades_of_gray": no_gray,
        "ablation_no_udcp": no_udcp,
        "ablation_no_clahe": no_clahe,
    }
    runtimes = {
        "raw": elapsed["raw"], "red_comp_only": elapsed["red_only"],
        "shades_of_gray_only": elapsed["shades_only"], "udcp_only": elapsed["udcp_only"],
        "clahe_only": elapsed["clahe_only"],
        "fixed_pipeline": elapsed["red_only"] + elapsed["shades_after_red"] + elapsed["udcp_after_color"] + elapsed["clahe_after_udcp"],
        "adaptive_pipeline": elapsed["adaptive"],
        "ablation_no_red_comp": elapsed["shades_without_red"] + elapsed["udcp_without_red"] + elapsed["clahe_without_red"],
        "ablation_no_shades_of_gray": elapsed["red_only"] + elapsed["udcp_without_shades"] + elapsed["clahe_without_shades"],
        "ablation_no_udcp": elapsed["red_only"] + elapsed["shades_after_red"] + elapsed["clahe_without_udcp"],
        "ablation_no_clahe": elapsed["red_only"] + elapsed["shades_after_red"] + elapsed["udcp_after_color"],
    }
    return methods, details, runtimes


def score_methods(methods, reference):
    result = {}
    ref_lab = rgb2lab(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    for name, image in methods.items():
        image_lab = rgb2lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result[name] = {
            "PSNR": psnr(image, reference),
            "SSIM": ssim(image, reference),
            "UIQM": uiqm(image),
            "UCIQE": uciqe(image),
            "CIEDE2000": float(np.mean(deltaE_ciede2000(image_lab, ref_lab))),
        }
    return result


def composite(psnr_value, ssim_value):
    return 0.5 * (float(np.clip(psnr_value / 40.0, 0, 1)) +
                  float(np.clip(ssim_value, 0, 1)))


def panel(image, title, size=256):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (size, 25), (0, 0, 0), -1)
    cv2.putText(canvas, title[:30], (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def paired_tests(test_rows):
    by_method = {}
    for row in test_rows:
        by_method.setdefault(row["method"], {})[row["image"]] = row
    fixed, adaptive = by_method["fixed_pipeline"], by_method["adaptive_pipeline"]
    common = sorted(set(fixed) & set(adaptive))
    metric_rows = []
    for metric in ("composite", "PSNR", "SSIM", "UIQM", "UCIQE"):
        a = np.array([adaptive[n][metric] for n in common], dtype=float)
        b = np.array([fixed[n][metric] for n in common], dtype=float)
        wins = int(np.sum(a > b))
        ties = int(np.sum(a == b))
        try:
            stat, pvalue = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            stat, pvalue = 0.0, 1.0
        metric_rows.append({"metric": metric, "n_pairs": len(common),
                            "adaptive_wins": wins, "ties": ties,
                            "win_rate_percent": 100.0 * wins / max(1, len(common)),
                            "adaptive_mean": float(a.mean()), "fixed_mean": float(b.mean()),
                            "mean_paired_delta": float((a - b).mean()),
                            "wilcoxon_statistic": float(stat), "wilcoxon_p_two_sided": float(pvalue)})
    return metric_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--reference", type=Path, default=REF_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--search-candidates", type=int, default=8,
                        help="Number of deterministic random grid candidates, plus current defaults.")
    parser.add_argument("--max-tune", type=int, default=None,
                        help="Development-only cap; do not use for final reported evaluation.")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Development-only cap; do not use for final reported evaluation.")
    args = parser.parse_args()
    if not 0.5 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0.5 and 1")
    if args.search_candidates < 1:
        parser.error("--search-candidates must be positive")
    if not args.input.is_dir() or not args.reference.is_dir():
        parser.error("Input and reference directories must exist")

    pairs, excluded = pair_files(args.input, args.reference)
    if len(pairs) < 4:
        parser.error("At least four readable, same-shape pairs are required")
    tune_pairs, test_pairs = split_pairs(pairs, args.train_fraction, args.seed)
    if args.max_tune is not None:
        tune_pairs = tune_pairs[:args.max_tune]
    if args.max_test is not None:
        test_pairs = test_pairs[:args.max_test]

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_configs(args.search_candidates, args.seed)
    print(f"Pairs={len(pairs)}; tuning={len(tune_pairs)}; test={len(test_pairs)}; "
          f"candidates={len(candidates)}; seed={args.seed}", flush=True)
    best_controller, search_rows = tune(tune_pairs, candidates)
    (args.output / "tuning_search.json").write_text(
        json.dumps({"objective": "mean of 0.5*SSIM + 0.5*clip(PSNR/40dB,0,1)",
                    "selected_controller": best_controller,
                    "candidates": search_rows}, indent=2), encoding="utf-8")

    split_manifest = {
        "research_question": "Does a degradation-aware controller improve a classical fixed pipeline on held-out paired images?",
        "seed": args.seed, "train_fraction": args.train_fraction,
        "tuning_images": [p.name for p, _ in tune_pairs],
        "test_images": [p.name for p, _ in test_pairs],
        "excluded_pairs": excluded,
        "full_reference_alignment": "Same filename stem and image dimensions; geometric alignment not independently verified.",
        "development_caps": {"max_tune": args.max_tune, "max_test": args.max_test},
    }
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    detail_rows, indicator_rows = [], []
    failure_dir = args.output / "worst_case_grids"
    failure_dir.mkdir(exist_ok=True)
    for index, (source, refpath) in enumerate(test_pairs, 1):
        image, reference = cv2.imread(str(source)), cv2.imread(str(refpath))
        methods, details, runtimes = run_variants(image, best_controller)
        scores = score_methods(methods, reference)
        image_rows = {}
        for method, values in scores.items():
            row = {"image": source.name, "method": method,
                   "runtime_seconds": runtimes[method], **values}
            row["composite"] = composite(values["PSNR"], values["SSIM"])
            detail_rows.append(row)
            image_rows[method] = row
        dg = details["degradation"]
        fixed_score = image_rows["fixed_pipeline"]["composite"]
        adaptive_score = image_rows["adaptive_pipeline"]["composite"]
        indicator_rows.append({"image": source.name, **dg,
                               "fixed_composite": fixed_score,
                               "adaptive_composite": adaptive_score,
                               "improvement": adaptive_score - fixed_score,
                               "red_alpha": details["parameters"]["red_alpha"],
                               "udcp_omega": details["parameters"]["udcp_omega"],
                               "clahe_clip_limit": details["parameters"]["clahe_clip_limit"]})
        if index % 10 == 0 or index == len(test_pairs):
            print(f"Test {index}/{len(test_pairs)}", flush=True)

    write_csv(args.output / "test_per_image.csv", detail_rows)
    write_csv(args.output / "indicator_per_image.csv", indicator_rows)

    methods = sorted({r["method"] for r in detail_rows})
    summary = []
    for method in methods:
        subset = [r for r in detail_rows if r["method"] == method]
        row = {"method": method, "n_images": len(subset)}
        for metric in ("runtime_seconds", "PSNR", "SSIM", "UIQM", "UCIQE", "CIEDE2000", "composite"):
            vals = [float(r[metric]) for r in subset]
            row[f"{metric}_mean"] = statistics.mean(vals)
            row[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(row)
    write_csv(args.output / "test_summary.csv", summary)
    write_csv(args.output / "paired_wilcoxon.csv", paired_tests(detail_rows))

    ablation_rows = [r for r in detail_rows if r["method"].startswith("ablation_")]
    write_csv(args.output / "ablation_per_image.csv", ablation_rows)

    corr_rows = []
    for indicator in ("red_deficiency", "haze_indicator", "contrast_deficiency"):
        x = np.array([r[indicator] for r in indicator_rows], dtype=float)
        y = np.array([r["improvement"] for r in indicator_rows], dtype=float)
        if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
        else:
            pr = pp = sr = sp = float("nan")
        corr_rows.append({"indicator": indicator, "n": len(x), "pearson_r": pr,
                          "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp})
    write_csv(args.output / "indicator_correlations.csv", corr_rows)

    try:
        import matplotlib.pyplot as plt
        for indicator in ("red_deficiency", "haze_indicator", "contrast_deficiency"):
            plt.figure(figsize=(6, 4))
            plt.scatter([r[indicator] for r in indicator_rows],
                        [r["improvement"] for r in indicator_rows], alpha=0.65)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.xlabel(indicator)
            plt.ylabel("Adaptive minus fixed composite score")
            plt.title(f"{indicator} vs held-out improvement")
            plt.tight_layout()
            plt.savefig(args.output / f"indicator_{indicator}.png", dpi=160)
            plt.close()
    except ImportError:
        print("matplotlib unavailable; correlation CSV was written without plots.")

    worst = sorted(indicator_rows, key=lambda r: r["improvement"])[:5]
    worst_rows = []
    for item in worst:
        image = cv2.imread(str(args.input / item["image"]))
        refmatch = next(r for r in test_pairs if r[0].name == item["image"])[1]
        reference = cv2.imread(str(refmatch))
        variants, _, _ = run_variants(image, best_controller)
        tiles = [panel(variants[k], title) for k, title in
                 (("raw", "Original"), ("fixed_pipeline", "Fixed"),
                  ("adaptive_pipeline", "Adaptive"))]
        if reference is not None:
            tiles.insert(1, panel(reference, "Reference"))
        cv2.imwrite(str(failure_dir / f"{Path(item['image']).stem}_comparison.png"),
                    np.hstack(tiles))
        worst_rows.append({**item, "review_note":
                           "Inspect saved comparison grid; metric deltas alone do not identify the visual failure cause."})
    write_csv(args.output / "worst_cases.csv", worst_rows)
    print(f"Completed held-out evaluation. Results: {args.output}")


if __name__ == "__main__":
    main()
