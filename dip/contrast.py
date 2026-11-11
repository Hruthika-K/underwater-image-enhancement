"""
Contrast enhancement for underwater images.
Problem: scattering makes images look hazy and flat (low contrast).
All functions take and return uint8 BGR images (OpenCV's default format).
Most work on the lightness (L) channel only, so colors are not distorted.
"""
import numpy as np             # array maths on pixel values
import cv2                     # OpenCV, used for color conversion and equalization


def _apply_to_lightness(img, func):        # helper: run a single-channel function on L only
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert BGR to LAB (L = lightness, A/B = color)
    l, a, b = cv2.split(lab)               # separate the three LAB channels
    l = func(l)                            # enhance only the lightness channel
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)  # recombine and convert back to BGR


def hist_equalization(img):                # method 1: spread pixel values evenly across 0-255
    return _apply_to_lightness(img, cv2.equalizeHist)  # global equalization on L (can over-amplify noise)


def clahe(img, clip_limit=2.0, tile_size=8):  # method 2: equalization done per tile, with a limit on boosting
    c = cv2.createCLAHE(clipLimit=clip_limit,  # clip_limit caps how much any histogram bin is boosted (less noise)
                        tileGridSize=(tile_size, tile_size))  # image is split into tile_size x tile_size regions
    return _apply_to_lightness(img, c.apply)   # apply CLAHE to the L channel only


def contrast_stretch(img, low=1.0, high=99.0):  # method 3: linearly stretch the useful range to 0-255
    def stretch(l):                        # inner function working on the single L channel
        lo, hi = np.percentile(l, [low, high])  # find the 1st and 99th percentile values (ignores outliers)
        if hi - lo < 1:                    # if the image is almost flat, stretching would just amplify noise
            return l                       # leave the channel unchanged
        out = (l.astype(np.float32) - lo) * 255.0 / (hi - lo)  # shift lo to 0 and scale hi to 255
        return np.clip(out, 0, 255).astype(np.uint8)  # clip values outside the range and return uint8
    return _apply_to_lightness(img, stretch)  # apply the stretch to the L channel only


def gamma_correction(img, gamma=1.2):      # method 4: brighten (gamma > 1) or darken (gamma < 1) mid-tones
    table = (np.arange(256) / 255.0) ** (1.0 / gamma) * 255  # precompute the output for every possible value 0-255
    return cv2.LUT(img, table.astype(np.uint8))  # look each pixel up in the table (fast, applied to all channels)


METHODS = {                                # lookup table so other scripts can pick a method by name
    "hist_eq": hist_equalization,
    "clahe": clahe,
    "stretch": contrast_stretch,
    "gamma": gamma_correction,
}


def enhance_contrast(img, method="clahe", **kwargs):  # one entry point: enhance_contrast(img, "stretch")
    if method not in METHODS:              # guard against typos in the method name
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")  # list the valid options
    return METHODS[method](img, **kwargs)  # call the chosen function, passing extra settings through