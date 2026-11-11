"""
Dehazing for underwater images.
Problem: light scatters off particles in the water, adding a veil of "haze" over the scene.
Model:   I = J * t + A * (1 - t)
         I = observed image, J = clean scene (what we want), t = transmission (how much scene light
         survives, 0-1), A = atmospheric / veiling light (the color of the haze itself).
All functions take and return uint8 BGR images (OpenCV's default format).
"""
import numpy as np             # array maths on pixel values
import cv2                     # OpenCV, used for min/box filters


def dark_channel(img_f, patch=15, channels=None):  # darkest value in each local patch
    sub = img_f if channels is None else img_f[..., channels]  # optionally use only some channels
    per_pixel_min = sub.min(axis=2)                # smallest value across the chosen channels at each pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))  # square patch window
    return cv2.erode(per_pixel_min, kernel)        # erode = minimum filter: darkest value in the patch


def estimate_atmospheric_light(img_f, dark, top_percent=0.1):
    """Estimate veiling light while avoiding bright, highly textured foregrounds."""
    n = max(1, int(dark.size * top_percent / 100))
    flat = img_f.reshape(-1, 3)
    gray = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F)).reshape(-1)
    blue_red = flat[:, 0] - flat[:, 2]
    score = dark.reshape(-1) + 0.15 * blue_red - 0.15 * _normalize_map(texture)
    idx = np.argpartition(score, -n)[-n:]
    candidates = flat[idx]
    a = candidates[np.argmax(candidates.sum(axis=1))]
    return np.maximum(a, 0.05)


def _normalize_map(value):
    low, high = np.percentile(value, [5, 95])
    return np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def guided_filter(guide, src, radius=30, eps=1e-3):  # edge-preserving smoothing (He et al. guided filter)
    size = (2 * radius + 1, 2 * radius + 1)         # box window size
    box = lambda x: cv2.boxFilter(x, -1, size)      # local average over that window
    mean_i, mean_p = box(guide), box(src)           # local averages of guide and source
    var_i = box(guide * guide) - mean_i * mean_i    # local variance of the guide
    cov_ip = box(guide * src) - mean_i * mean_p     # local covariance of guide and source
    a = cov_ip / (var_i + eps)                      # slope: eps controls how much smoothing happens
    b = mean_p - a * mean_i                         # intercept
    return box(a) * guide + box(b)                  # smoothed result that follows the guide's edges


def dehaze(img, method="udcp", patch=15, omega=0.95, t0=0.25, refine=True, return_t=False):
    """Dehaze with DCP, UDCP, or a red-channel-aware prior."""
    f = img.astype(np.float32) / 255.0              # work in 0-1 floats
    channels = None if method == "dcp" else [0, 1]
    if method not in ("dcp", "udcp", "red"):
        raise ValueError("method must be 'dcp', 'udcp', or 'red'")

    dark = dark_channel(f, patch, channels)         # dark channel of the observed image
    a = estimate_atmospheric_light(f, dark)         # haze color A, one value per channel [B, G, R]

    normed = f / a
    if method == "red":
        red = cv2.erode(normed[..., 2], cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch, patch)
        ))
        t = 1.0 - omega * red
    else:
        t = 1.0 - omega * dark_channel(normed, patch, channels)

    if refine:                                      # raw t is blocky because of the patches
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0  # guide image
        t = np.clip(guided_filter(gray, t.astype(np.float32)), 0, 1)  # smooth t while keeping object edges

    t_safe = np.maximum(t, t0)[..., None]
    channel_scale = np.array([1.0, 1.0, 0.85], dtype=np.float32)
    channel_t = np.maximum(t_safe * channel_scale, t0)
    j = (f - a) / channel_t + a
    out = (np.clip(j, 0, 1) * 255).round().astype(np.uint8)  # clip to valid range, back to uint8
    return (out, t) if return_t else out            # optionally return the transmission map too