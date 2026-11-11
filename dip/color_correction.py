"""
Color correction for underwater images.
Problem: water absorbs red light first, so images look blue/green.
All functions take and return uint8 BGR images (OpenCV's default format).
"""
import numpy as np             # array maths on pixel values
import cv2                     # OpenCV, used here to split/merge color channels


def gray_world(img, max_gain=1.8, strength=0.4):  # method 1: partial gray-world correction
    if max_gain <= 0:
        raise ValueError("max_gain must be positive")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    f = img.astype(np.float32)             # convert to float so multiplying doesn't overflow uint8
    means = f.reshape(-1, 3).mean(axis=0)  # flatten to (pixels, 3) and average each channel -> [B, G, R] means
    gains = means.mean() / (means + 1e-6)  # gain per channel = overall mean / channel mean (1e-6 avoids divide by zero)
    gains = np.clip(gains, 1.0 / max_gain, max_gain)  # cap channel correction
    gains = 1.0 + strength * (gains - 1.0)
    out = f * gains                        # scale each channel by its gain (broadcasts over H x W)
    return np.clip(out, 0, 255).astype(np.uint8)  # keep values in 0-255 and go back to uint8


def white_patch(img, percentile=99.0, max_gain=3.0):  # method 2: assume the brightest pixels should be white
    f = img.astype(np.float32)             # convert to float for safe multiplication
    ref = np.percentile(f.reshape(-1, 3), percentile, axis=0)  # per-channel "white" level (99th percentile ignores outliers)
    gains = 255.0 / (ref + 1e-6)           # gain that would push each channel's white level up to 255
    gains = np.clip(gains, 1.0, max_gain)  # never darken, and cap the boost
    out = f * gains                        # apply the per-channel gains
    return np.clip(out, 0, 255).astype(np.uint8)  # clip to valid range and return uint8


def shades_of_gray(
    img,
    power=6.0,
    max_gain=2.2,
    low=1.0,
    high=99.0,
    stretch=True,
):
    """Balance channels using Minkowski means and robust percentile limits."""
    if power <= 0 or max_gain <= 0 or low < 0 or high > 100 or low >= high:
        raise ValueError("invalid Shades-of-Gray parameters")
    f = img.astype(np.float32) / 255.0
    means = np.mean(np.power(f, power), axis=(0, 1)) ** (1.0 / power)
    gains = np.mean(means) / (means + 1e-6)
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    balanced = np.clip(f * gains, 0.0, 1.0)
    if not stretch:
        return (balanced * 255).round().astype(np.uint8)
    limits = np.percentile(balanced.reshape(-1, 3), [low, high], axis=0)
    stretched = (balanced - limits[0]) / np.maximum(limits[1] - limits[0], 1e-6)
    return (np.clip(stretched, 0.0, 1.0) * 255).round().astype(np.uint8)


def red_channel_compensation(img, alpha=1.0, reference="green"):
    """Rebuild lost red using green or a blue/green average as reference."""
    if reference not in ("green", "blue_green"):
        raise ValueError("reference must be 'green' or 'blue_green'")
    f = img.astype(np.float32) / 255.0     # scale to 0-1, the range this formula is written for
    b, g, r = cv2.split(f)                 # separate the three channels (OpenCV order is B, G, R)
    # add back red in proportion to how much weaker red is than green (g.mean() - r.mean()),
    # weighted by (1 - r) so already-red pixels aren't oversaturated, and by g so it follows scene structure
    reference_mean = (
        g.mean() if reference == "green" else (g.mean() + b.mean()) / 2.0
    )
    r_comp = r + alpha * (reference_mean - r.mean()) * (1.0 - r) * g
    r_comp = np.clip(r_comp, 0.0, 1.0)     # keep the new red channel within 0-1
    out = cv2.merge([b, g, r_comp])        # put the channels back together, with the corrected red
    return (out * 255).round().astype(np.uint8)  # scale back to 0-255 and convert to uint8


METHODS = {                                # lookup table so other scripts can pick a method by name
    "gray_world": gray_world,
    "white_patch": white_patch,
    "shades_of_gray": shades_of_gray,
    "red_comp": red_channel_compensation,
}


def correct(img, method="gray_world", **kwargs):  # one entry point: correct(img, "white_patch")
    if method not in METHODS:              # guard against typos in the method name
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")  # tell the user valid options
    return METHODS[method](img, **kwargs)  # call the chosen function, passing any extra settings through

def vibrance(img, amount=0.6):
    """Boost saturation more on dull pixels than on already-vivid ones."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    hsv[..., 1] = np.clip(s * (1.0 + amount * (1.0 - s)), 0.0, 1.0) * 255.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

METHODS["vibrance"] = vibrance