"""Learned per-image pipeline selection vs. one fixed pipeline (same split as before).

Question: can a model trained on cheap image features pick, per image, a better
pipeline than always using the best single pipeline?

Steps
  1. Score every pipeline in a small POOL on ALL pairs (cached, resumable).
  2. Oracle headroom: per-image best pipeline vs. best single pipeline (test split).
  3. Train a selector ONLY on the tuning split (model chosen by 5-fold CV there).
  4. Evaluate the selector ONCE on the held-out test split; paired Wilcoxon +
     bootstrap CI against the best single pipeline (chosen on tuning).

Needs results/candidate_pipelines/selected_configs.json from candidate_pipelines.py
(pass --configs to point at your selected_configs_1.json). Place next to evaluate_all.py.

    python selector_experiment.py --configs selected_configs_1.json
    python selector_experiment.py --configs selected_configs_1.json --limit 20   # smoke test
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab

from candidate_pipelines import PIPES, compare, metrics_of
from evaluate_all import RAW_DEFAULT, REF_DEFAULT, pair_files, split_pairs, write_csv

OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "selector"
METRICS = ("PSNR", "SSIM", "CIEDE2000", "UIQM", "balanced")


def build_pool(configs_path):
    w = json.loads(Path(configs_path).read_text(encoding="utf-8"))["winners"]
    pool = {
        "A_sog": ("A_sog", w["A_sog"]["params"]),
        "B_sog_clahe": ("B_sog_clahe", w["B_sog_clahe"]["params"]),
        "C_best": ("C_red_sog_clahe", w["C_red_sog_clahe"]["params"]),
        "D_weak_dehaze": ("D_weak_dehaze", w["D_weak_dehaze"]["params"]),
    }
    c = dict(w["C_red_sog_clahe"]["params"])
    pool["C_clip2"] = ("C_red_sog_clahe", {**c, "clip": 2.0})
    pool["C_alpha0.2"] = ("C_red_sog_clahe", {**c, "alpha": 0.2})
    return pool


# ------------------------------------------------------------------ features
def features(img):
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    p1, p99 = np.percentile(L, [1, 99])
    m = cv2.boxFilter(L, -1, (15, 15))
    local_std = np.sqrt(np.maximum(cv2.boxFilter(L * L, -1, (15, 15)) - m * m, 0))
    hist = np.histogram(L, bins=64, range=(0, 1))[0].astype(np.float64)
    pr = hist / max(hist.sum(), 1.0)
    entropy = float(-(pr[pr > 0] * np.log2(pr[pr > 0])).sum())
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return {"mean_b": mb, "mean_g": mg, "mean_r": mr,
            "std_b": float(b.std()), "std_g": float(g.std()), "std_r": float(r.std()),
            "r_over_g": mr / max(mg, 1e-6), "b_over_g": mb / max(mg, 1e-6),
            "red_def": (mg - mr) / max(mg, 1e-6),
            "sat_mean": float(hsv[..., 1].mean()), "sat_std": float(hsv[..., 1].std()),
            "val_mean": float(hsv[..., 2].mean()),
            "L_mean": float(L.mean()), "L_std": float(L.std()),
            "L_p1": float(p1), "L_p99": float(p99), "L_spread": float(p99 - p1),
            "dark_mean": float(f.min(axis=2).mean()),
            "local_contrast": float(local_std.mean()), "L_entropy": entropy,
            "grad_mean": float(grad.mean())}


# ----------------------------------------------------------- cached scoring
def compute_all(pairs, pool, out_dir):
    sc_path, ft_path = out_dir / "scores_all.csv", out_dir / "features_all.csv"
    done = set()
    if sc_path.exists():
        with open(sc_path, newline="", encoding="utf-8") as fh:
            done = {r["image"] for r in csv.DictReader(fh)}
    todo = [(s, r) for s, r in pairs if s.name not in done]
    print(f"Scoring: {len(pairs) - len(todo)} cached, {len(todo)} to do", flush=True)
    sc_new, ft_new = not sc_path.exists(), not ft_path.exists()
    with open(sc_path, "a", newline="", encoding="utf-8") as sf, \
            open(ft_path, "a", newline="", encoding="utf-8") as ff:
        sw = ftw = None
        for i, (src, refp) in enumerate(todo, 1):
            img, ref = cv2.imread(str(src)), cv2.imread(str(refp))
            ref_lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
            if ftw is None:
                ftw = csv.DictWriter(ff, ["image"] + list(features(img)))
                if ft_new:
                    ftw.writeheader()
            ftw.writerow({"image": src.name, **features(img)})
            for name, (fam, p) in pool.items():
                m = metrics_of(PIPES[fam](img, p), ref, ref_lab, with_uciqe=False)
                row = {"image": src.name, "pipeline": name,
                       **{k: m[k] for k in METRICS}}
                if sw is None:
                    sw = csv.DictWriter(sf, list(row))
                    if sc_new:
                        sw.writeheader()
                sw.writerow(row)
            if i % 10 == 0:
                sf.flush(); ff.flush()
                print(f"  scored {i}/{len(todo)}", flush=True)


def load_tables(names, pipe_names):
    with open(out_scores := names[0], newline="", encoding="utf-8") as fh:
        scores = list(csv.DictReader(fh))
    with open(names[1], newline="", encoding="utf-8") as fh:
        feats = {r["image"]: r for r in csv.DictReader(fh)}
    return scores, feats


# ------------------------------------------------------------------- models
class Ridge:
    def __init__(self, alpha): self.alpha = alpha

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        self.ym = Y.mean(0)
        self.W = np.linalg.solve(Xs.T @ Xs + self.alpha * np.eye(X.shape[1]), Xs.T @ (Y - self.ym))
        return self

    def predict(self, X): return ((X - self.mu) / self.sd) @ self.W + self.ym


class GBR:
    def fit(self, X, Y):
        from sklearn.ensemble import GradientBoostingRegressor
        self.ms = [GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                             subsample=0.8, random_state=0).fit(X, Y[:, j])
                   for j in range(Y.shape[1])]
        return self

    def predict(self, X): return np.column_stack([m.predict(X) for m in self.ms])


def model_zoo():
    zoo = {f"ridge_{a}": (lambda a=a: Ridge(a)) for a in (1, 10, 100, 1000)}
    try:
        import sklearn  # noqa: F401
        zoo["gbr"] = GBR
    except ImportError:
        print("scikit-learn not found: using ridge models only")
    return zoo


def cv_score(make, X, Y, seed, folds=5):
    n = len(X)
    folds = max(2, min(folds, n))
    idx = np.random.default_rng(seed).permutation(n)
    parts, vals = np.array_split(idx, folds), []
    for k in range(folds):
        te, tr = parts[k], np.concatenate([parts[j] for j in range(folds) if j != k])
        pred = make().fit(X[tr], Y[tr]).predict(X[te])
        vals.append(Y[te][np.arange(len(te)), pred.argmax(1)].mean())
    return float(np.mean(vals))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--reference", type=Path, default=REF_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--limit", type=int, default=None, help="Development: use first N pairs only.")
    ap.add_argument("--tune-n", type=int, default=None,
                    help="Use only the first N tuning images (test split stays full). Faster.")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="Pool pipelines to skip, e.g. --drop D_weak_dehaze (slowest). "
                         "A_sog and C_best cannot be dropped.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pairs, _ = pair_files(args.input, args.reference)
    tune_pairs, test_pairs = split_pairs(pairs, args.train_fraction, args.seed)
    if args.tune_n:
        tune_pairs = tune_pairs[:args.tune_n]
    if args.limit:
        tune_pairs, test_pairs = tune_pairs[:args.limit], test_pairs[:max(3, args.limit // 2)]
    pool = build_pool(args.configs)
    for d in args.drop:
        if d in ("A_sog", "C_best"):
            raise SystemExit(f"{d} is required and cannot be dropped")
        pool.pop(d, None)
    names = list(pool)
    compute_all(tune_pairs + test_pairs, pool, args.output)

    scores, feats = load_tables((args.output / "scores_all.csv", args.output / "features_all.csv"), names)
    table = {(r["image"], r["pipeline"]): r for r in scores}
    fkeys = [k for k in next(iter(feats.values())) if k != "image"]

    def mats(pr):
        imgs = [s.name for s, _ in pr]
        X = np.array([[float(feats[n][k]) for k in fkeys] for n in imgs])
        M = {m: np.array([[float(table[(n, p)][m]) for p in names] for n in imgs]) for m in METRICS}
        return imgs, X, M

    tr_imgs, Xtr, Mtr = mats(tune_pairs)
    te_imgs, Xte, Mte = mats(test_pairs)
    Ytr, Yte = Mtr["balanced"], Mte["balanced"]

    # best single pipeline, chosen on the tuning split
    best_single = int(Ytr.mean(0).argmax())
    # model chosen by CV on the tuning split
    zoo = model_zoo()
    cv = {k: cv_score(mk, Xtr, Ytr, args.seed) for k, mk in zoo.items()}
    best_model = max(cv, key=cv.get)
    cv_single = float(Ytr[:, best_single].mean())
    print("CV (tuning) achieved balanced:", {k: round(v, 4) for k, v in cv.items()},
          "| best single on tuning:", round(cv_single, 4))
    sel = zoo[best_model]().fit(Xtr, Ytr)
    choice = sel.predict(Xte).argmax(1)

    rows = []
    def add(method, idx):
        for i, n in enumerate(te_imgs):
            rows.append({"image": n, "method": method,
                         **{m: Mte[m][i, idx[i]] for m in METRICS}})
    add("selector", choice)
    add(f"always_{names[best_single]}", np.full(len(te_imgs), best_single))
    add("oracle", Yte.argmax(1))
    add("sog_default_like_A", np.full(len(te_imgs), names.index("A_sog")))
    write_csv(args.output / "selector_test_per_image.csv", rows)

    comps = compare(rows, "selector", f"always_{names[best_single]}", args.seed)
    comps += compare(rows, "oracle", f"always_{names[best_single]}", args.seed)
    write_csv(args.output / "selector_comparisons.csv", comps)

    summary = {"n_tune": len(tr_imgs), "n_test": len(te_imgs), "pool": names,
               "best_single_on_tuning": names[best_single], "selected_model": best_model,
               "cv_achieved_balanced_tuning": cv, "tuning_best_single_balanced": cv_single,
               "test_balanced": {"selector": float(Yte[np.arange(len(Yte)), choice].mean()),
                                 f"always_{names[best_single]}": float(Yte[:, best_single].mean()),
                                 "oracle": float(Yte.max(1).mean()),
                                 "random_choice_expected": float(Yte.mean())},
               "selector_choice_counts": {n: int((choice == j).sum()) for j, n in enumerate(names)},
               "oracle_choice_counts": {n: int((Yte.argmax(1) == j).sum()) for j, n in enumerate(names)}}
    (args.output / "selector_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["test_balanced"], indent=2))
    print("Selector choices:", summary["selector_choice_counts"])
    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()