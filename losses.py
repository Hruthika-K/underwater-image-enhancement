"""
Composite loss for underwater image enhancement.

Design (all terms roughly in [0, 1], so weights are meaningful):
  L = w_l1  * L1(img, ref)            pixel fidelity (robust to outliers)
    + w_ss  * (1 - SSIM(img, ref))    structural fidelity
    + w_gr  * grad_L1(img, ref)       edge fidelity (dehazing tends to blur edges)
    + w_lab * lab_loss(img, ref)      perceptual color error in CIELab
    + w_uq  * uciqe_hinge(img, ref)   no-ref safeguard: output must not score
                                      worse than the reference on UCIQE

Why the UCIQE hinge: a pure L1 loss can be gamed by a washed-out image that
stays close to the hazy reference. UCIQE (Yang & Sowmya, 2015) rewards real
chroma spread, luminance contrast and saturation, so the hinge pushes the
output to genuinely look "cleaner underwater" than the target.

No-reference mode (ref=None) keeps only UCIQE/UIQM-based terms, e.g. to score
or optimize enhancement without ground truth.

Everything here is plain NumPy/OpenCV (matches the repo's dependencies), but
every term has a direct differentiable PyTorch equivalent (see the note at the
bottom of this docstring), so the same objective can train a network.

PyTorch port note: replace np.sort-based percentiles inside UCIQE with a soft
quantile (e.g. differentiable sorting or a sigmoid-smoothed percentile), and
use kornia/naive conv implementations for SSIM and Sobel. The cv2 calls here
(ssim via skimage, Sobel via cv2) mirror those ops.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from dip.color_correction import correct
from dip.dehazing import dehaze
from dip.contrast import enhance_contrast
from metrics import uciqe, uiqm

# default term weights (see module docstring)
WEIGHTS = {
    "l1": 1.0,
    "ssim": 0.5,
    "grad": 0.5,
    "lab": 0.5,
    "uciqe_hinge": 1.0,  # normalized to 0-1 inside the function
}


def l1_loss(img, ref):
    return float(np.abs(img.astype(np.float32) - ref.astype(np.float32)).mean() / 255.0)  # scale to 0-1


def dssim_loss(img, ref):
    return float(1.0 - structural_similarity(ref, img, channel_axis=2, data_range=255))  # 0 = identical


def grad_loss(img, ref):
    """L1 between Sobel edge-magnitude maps, normalized to 0-1."""

    def edges(x):
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    diff = np.abs(edges(img) - edges(ref))          # per-pixel edge error
    return float(diff.mean() / 255.0)               # same 0-1 scaling as L1


def lab_loss(img, ref):
    """Mean absolute error in CIELab, normalized to 0-1.

    Catches perceptual color casts: L1 in BGR underweights a red-channel error
    compared to how visibly wrong it looks to a human.
    """
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)  # L 0-100, a/b 0-255 (offset)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    a[..., 0] /= 100.0                              # L to 0-1
    b[..., 0] /= 100.0
    a[..., 1:] = (a[..., 1:] - 128.0) / 127.0       # a/b to roughly -1..1
    b[..., 1:] = (b[..., 1:] - 128.0) / 127.0
    return float(np.abs(a - b).mean())


def uciqe_hinge(img, ref):
    """Penalize only when the output is worse than the reference on UCIQE.

    metrics.uciqe returns 0-100-scale values, so divide by 100 to put the
    hinge on the same 0-1 footing as the other terms.
    """
    return float(max(0.0, uciqe(ref) - uciqe(img))) / 100.0  # 0 when output >= reference


def total_loss(img, ref, weights=None):
    """Composite loss. img/ref are uint8 BGR, same shape."""
    if img.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {img.shape} vs {ref.shape}. Resize first.")

    w = {**WEIGHTS, **(weights or {})}

    parts = {
        "l1": l1_loss(img, ref),
        "ssim": dssim_loss(img, ref),
        "grad": grad_loss(img, ref),
        "lab": lab_loss(img, ref),
        "uciqe_hinge": uciqe_hinge(img, ref),
    }

    total = sum(w[k] * v for k, v in parts.items())
    return total, parts


def total_loss_noref(img):
    """No-reference variant: only underwater quality priors (lower = better).

    Negative UCIQE/UIQM because higher metric values mean better images.
    Scale-normalized so the two terms are comparable in magnitude.
    """
    u, q = uciqe(img), uiqm(img)
    return -u - q / 10.0, {"uciqe": u, "uiqm": q}


# ---------------------------------------------------------------- demo / eval

ROOT = Path(__file__).resolve().parent


def enhance(img):
    """Same 5-step pipeline as simple_enhance.py, kept in sync on purpose."""
    x = correct(img, "red_comp", alpha=0.60, reference="green")
    x = correct(x, "shades_of_gray", max_gain=1.45, stretch=False)
    dehazed = dehaze(x, method="udcp", omega=0.72, t0=0.38)
    x = cv2.addWeighted(x, 0.55, dehazed, 0.45, 0)
    return enhance_contrast(x, method="clahe", clip_limit=1.8, tile_size=8)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the composite loss.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=str, default=str(ROOT / "raw-890"))
    parser.add_argument("--reference", type=str, default=str(ROOT / "reference-890"))
    args = parser.parse_args()

    raw_dir, ref_dir = Path(args.input), Path(args.reference)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    images = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in exts)
    random.seed(args.seed)
    selected = random.sample(images, min(args.n, len(images)))

    terms = list(WEIGHTS)
    header = f"{'image':>14} | " + " | ".join(f"{t:>11}" for t in terms) + " |    total"
    print(header)
    print("-" * len(header))

    rows, raw_rows = [], []
    for path in selected:
        ref_path = ref_dir / path.name
        raw, ref = cv2.imread(str(path)), cv2.imread(str(ref_path))
        if raw is None or ref is None:
            print(f"{path.name:>14} | unreadable pair, skipped")
            continue

        enhanced = enhance(raw)
        total, parts = total_loss(enhanced, ref)
        raw_total, _ = total_loss(raw, ref)

        rows.append((path.name, parts, total))
        raw_rows.append((path.name, raw_total))
        print(
            f"{path.name:>14} | "
            + " | ".join(f"{parts[t]:11.4f}" for t in terms)
            + f" | {total:8.4f}"
        )

    print("-" * len(header))
    mean_enhanced = float(np.mean([t for _, _, t in rows]))
    mean_raw = float(np.mean([t for _, t in raw_rows]))
    print(f"{'MEAN enhanced':>14} | " + " | ".join(
        f"{float(np.mean([p[t] for _, p, _ in rows])):11.4f}" for t in terms
    ) + f" | {mean_enhanced:8.4f}")
    print(f"{'MEAN raw':>14} | {'(pipeline baseline)':^59} | {mean_raw:8.4f}")
    print(f"\nLoss reduction from enhancement: {mean_raw - mean_enhanced:+.4f} "
          f"(positive = pipeline beats the raw input under this loss)")

    # distribution stats + CSV report for the whole sample
    if rows:
        enh = np.array([t for _, _, t in rows])
        raws = np.array([t for _, t in raw_rows])

        print("\nDistribution of total loss "
              f"({len(rows)} images, seed {args.seed}):")
        print(f"{'':>14} |    min |    25% | median |    75% |    max")
        for label, arr in [("enhanced", enh), ("raw", raws)]:
            q = np.percentile(arr, [0, 25, 50, 75, 100])
            print(f"{label:>14} | " + " | ".join(f"{v:6.4f}" for v in q))

        best_i, worst_i = int(np.argmin(enh)), int(np.argmax(enh))
        print(f"\nBest image under this loss : {rows[best_i][0]} (total {enh[best_i]:.4f})")
        print(f"Worst image under this loss: {rows[worst_i][0]} (total {enh[worst_i]:.4f})")

        import csv
        report_dir = ROOT / "results"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"loss_report_seed{args.seed}_n{len(rows)}.csv"
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "raw_total", "enhanced_total",
                             "improvement"] + terms)
            for (name, parts, total), (raw_name, raw_total) in zip(rows, raw_rows):
                writer.writerow(
                    [name, f"{raw_total:.6f}", f"{total:.6f}",
                     f"{raw_total - total:.6f}"] + [f"{parts[t]:.6f}" for t in terms]
                )
        print(f"\nPer-image report saved to: {report_path}")


if __name__ == "__main__":
    main()