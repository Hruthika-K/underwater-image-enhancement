"""
Image quality metrics.
  With a reference image : PSNR, SSIM            (higher = closer to the reference)
  Without a reference    : UIQM, UCIQE           (higher = better underwater image quality)
All functions take uint8 BGR images (OpenCV's default format).
"""
import numpy as np                                   # array maths
import cv2                                           # color conversion and Sobel edges
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # standard full-reference metrics
from skimage.color import rgb2lab                    # RGB -> CIELab conversion used by UCIQE


# ------------------------------------------------------------------ full-reference

def _check_same_shape(a, b):                         # helper: both images must be the same size
    if a.shape != b.shape:                           # compare height, width and channels
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}. Resize before scoring.")  # explain the problem


def psnr(img, ref):                                  # peak signal-to-noise ratio, in dB
    _check_same_shape(img, ref)                      # make sure the comparison is valid
    return float(peak_signal_noise_ratio(ref, img, data_range=255))  # data_range=255 because pixels are uint8


def ssim(img, ref):                                  # structural similarity, from 0 to 1
    _check_same_shape(img, ref)                      # make sure the comparison is valid
    return float(structural_similarity(ref, img, channel_axis=2, data_range=255))  # channel_axis=2: color image


# ------------------------------------------------------------------ UCIQE

def uciqe(img_bgr):                                  # underwater color image quality evaluation (Yang & Sowmya, 2015)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # skimage expects RGB order, not BGR
    lab = rgb2lab(rgb)                               # convert to CIELab (L is 0-100, a/b are roughly -128..127)
    l = lab[..., 0]                                  # lightness channel
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)  # color intensity = distance from gray in the a-b plane

    sigma_c = chroma.std()                           # term 1: spread of chroma (how varied the colors are)

    flat_l = np.sort(l, axis=None)                   # all lightness values sorted from dark to bright
    top = max(1, int(round(0.01 * flat_l.size)))     # number of pixels in 1% of the image (at least 1)
    con_l = flat_l[-top:].mean() - flat_l[:top].mean()  # term 2: brightest 1% minus darkest 1% = luminance contrast

    sat = np.where(l > 0, chroma / np.maximum(l, 1e-6), 0.0)  # saturation = chroma / lightness (0 where L is 0)
    mu_s = sat.mean()                                # term 3: average saturation

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)  # weighted sum, weights from the paper


# ------------------------------------------------------------------ UIQM

def _block_view(x, size):                            # helper: cut a 2D array into non-overlapping size x size blocks
    h, w = (x.shape[0] // size) * size, (x.shape[1] // size) * size  # trim so both sides divide evenly by size
    x = x[:h, :w]                                    # drop the leftover edge pixels
    return x.reshape(h // size, size, w // size, size)  # shape (block rows, size, block cols, size)


def _uicm(rgb):                                      # colorfulness measure
    r, g, b = [rgb[..., i].astype(np.float64) for i in range(3)]  # split channels as floats
    rg = (r - g).ravel()                             # red-green opponent channel, flattened
    yb = ((r + g) / 2 - b).ravel()                   # yellow-blue opponent channel, flattened

    def trimmed_stats(v, trim=0.1):                  # asymmetric alpha-trimmed mean and variance
        v = np.sort(v)                               # sort so the extremes sit at both ends
        k = v.size                                   # number of pixels
        lo, hi = int(np.ceil(trim * k)), int(k - np.floor(trim * k))  # indices that cut 10% off each end
        v = v[lo:hi]                                 # keep the middle 80% (removes outliers)
        return v.mean(), v.var()                     # mean and variance of what remains

    mu_rg, var_rg = trimmed_stats(rg)                # stats for the red-green channel
    mu_yb, var_yb = trimmed_stats(yb)                # stats for the yellow-blue channel
    return -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(var_rg + var_yb)  # combine, weights from paper


def _eme(x, size=8):                                 # enhancement measure estimation: log of max/min per block
    blocks = _block_view(x, size)                    # cut into 8x8 blocks
    mx = blocks.max(axis=(1, 3))                     # brightest value in each block
    mn = blocks.min(axis=(1, 3))                     # darkest value in each block
    valid = (mx > 0) & (mn > 0)                      # log is undefined when a block has a zero
    if not valid.any():                              # no usable blocks
        return 0.0                                   # nothing to measure
    return 2.0 * np.log(mx[valid] / mn[valid]).sum() / mx.size  # (2 / number of blocks) * sum of log ratios


def _uism(rgb):                                      # sharpness measure
    total = 0.0                                      # running weighted sum
    for i, weight in enumerate((0.299, 0.587, 0.114)):  # R, G, B weights (same as grayscale conversion)
        ch = rgb[..., i].astype(np.float64)          # one channel as float
        edge = np.hypot(cv2.Sobel(ch, cv2.CV_64F, 1, 0),   # horizontal Sobel gradient
                        cv2.Sobel(ch, cv2.CV_64F, 0, 1))   # vertical Sobel gradient, combined into edge strength
        total += weight * _eme(edge * ch)            # weight edges by the channel itself, then score with EME
    return total                                     # sharpness score


def _uiconm(rgb, size=8):                            # contrast measure (log-AMEE, simplified)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)  # work on brightness only
    blocks = _block_view(gray, size)                 # cut into 8x8 blocks
    mx = blocks.max(axis=(1, 3))                     # block maximum
    mn = blocks.min(axis=(1, 3))                     # block minimum
    top, bot = mx - mn, mx + mn                      # numerator and denominator of the contrast ratio
    valid = (top > 0) & (bot > 0)                    # skip flat blocks (ratio would be 0 or undefined)
    if not valid.any():                              # no usable blocks
        return 0.0                                   # nothing to measure
    ratio = top[valid] / bot[valid]                  # local contrast between 0 and 1
    return float(-(ratio * np.log(ratio)).sum() / mx.size)  # entropy-style sum, normalised by number of blocks


def uiqm(img_bgr):                                   # underwater image quality measure (Panetta et al., 2016)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # the sub-metrics are written for RGB order
    return float(0.0282 * _uicm(rgb)                 # colorfulness
                 + 0.2953 * _uism(rgb)               # sharpness
                 + 3.5753 * _uiconm(rgb))            # contrast, weights from the paper


# ------------------------------------------------------------------ convenience

def evaluate(img, ref=None):                         # compute every metric at once
    scores = {"UIQM": uiqm(img), "UCIQE": uciqe(img)}  # no-reference metrics always available
    if ref is not None:                              # only if a reference image was supplied
        scores["PSNR"] = psnr(img, ref)              # add PSNR
        scores["SSIM"] = ssim(img, ref)              # add SSIM
    return scores                                    # dictionary of metric name -> value