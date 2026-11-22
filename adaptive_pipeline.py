
"""
Degradation-aware adaptive underwater enhancement.

Uses existing project implementations:
1. color_correction.py
2. dehazing.py
3. contrast.py

Input/output: uint8 BGR images.
"""

import cv2
import numpy as np

from dip.color_correction import correct
from dip.dehazing import dehaze
from dip.contrast import enhance_contrast


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def estimate_degradation(image):
    """
    Estimate color deficiency, haze and local contrast.

    All indicators are heuristic proxies and require validation.
    """

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a valid 3-channel BGR image.")

    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 BGR image.")

    f = image.astype(np.float32) / 255.0

    # --------------------------------------------------
    # 1. RED CHANNEL DEFICIENCY
    # --------------------------------------------------

    blue_mean, green_mean, red_mean = f.mean(axis=(0, 1))

    red_deficiency = clip01(
        (green_mean - red_mean) / max(green_mean, 1e-6)
    )

    # --------------------------------------------------
    # 2. HAZE ESTIMATION USING UDCP TRANSMISSION
    # --------------------------------------------------

    _, transmission = dehaze(
        image,
        method="udcp",
        patch=15,
        omega=0.75,
        t0=0.25,
        refine=True,
        return_t=True,
    )

    mean_transmission = float(np.mean(transmission))
    median_transmission = float(np.median(transmission))

    haze_indicator = clip01(1.0 - mean_transmission)

    # --------------------------------------------------
    # 3. LOCAL CONTRAST ESTIMATION
    # --------------------------------------------------

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    luminance = lab[:, :, 0].astype(np.float32)

    local_mean = cv2.boxFilter(
        luminance,
        -1,
        (15, 15),
        normalize=True,
    )

    local_mean_sq = cv2.boxFilter(
        luminance * luminance,
        -1,
        (15, 15),
        normalize=True,
    )

    local_variance = np.maximum(
        local_mean_sq - local_mean * local_mean,
        0.0,
    )

    local_std = np.sqrt(local_variance)

    local_contrast = float(np.mean(local_std))

    # Initial reference threshold. Validate on your dataset.
    target_local_contrast = 25.0

    contrast_deficiency = clip01(
        (target_local_contrast - local_contrast)
        / target_local_contrast
    )

    # --------------------------------------------------
    # 4. HIGH-FREQUENCY RESPONSE
    # --------------------------------------------------

    median_filtered = cv2.medianBlur(
        luminance.astype(np.uint8),
        3,
    ).astype(np.float32)

    high_frequency_residual = np.abs(
        luminance - median_filtered
    )

    high_frequency_score = clip01(
        float(np.median(high_frequency_residual)) / 12.0
    )

    return {
        "red_deficiency": red_deficiency,
        "haze_indicator": haze_indicator,
        "contrast_deficiency": contrast_deficiency,
        "local_contrast": local_contrast,
        "high_frequency_score": high_frequency_score,
        "mean_transmission": mean_transmission,
        "median_transmission": median_transmission,
        "red_mean": float(red_mean),
        "green_mean": float(green_mean),
        "blue_mean": float(blue_mean),
    }


DEFAULT_CONTROLLER = {
    "red_alpha_base": 0.15,
    "red_alpha_slope": 0.25,
    "udcp_omega_base": 0.55,
    "udcp_omega_slope": 0.25,
    "clahe_slope": 0.8,
}


def select_adaptive_parameters(degradation, controller=None):
    """
    Conservative adaptive parameter controller.

    Initial parameter ranges must be tuned using validation data.
    """

    red_def = degradation["red_deficiency"]
    haze = degradation["haze_indicator"]
    contrast_def = degradation["contrast_deficiency"]
    high_frequency = degradation["high_frequency_score"]

    # Conservative red compensation.
    # Use green reference to avoid blue channel increasing reference mean.
    coefficients = {**DEFAULT_CONTROLLER, **(controller or {})}
    red_alpha = float(
        np.clip(
            coefficients["red_alpha_base"] + coefficients["red_alpha_slope"] * red_def,
            coefficients["red_alpha_base"],
            0.40,
        )
    )

    # Stronger haze estimate -> stronger UDCP.
    udcp_omega = float(
        np.clip(
            coefficients["udcp_omega_base"] + coefficients["udcp_omega_slope"] * haze,
            coefficients["udcp_omega_base"],
            0.80,
        )
    )

    # Fixed transmission floor.
    udcp_t0 = 0.25

    # Increase CLAHE only when local contrast is low.
    # Reduce enhancement when high-frequency response is high.
    clahe_clip_limit = float(
        np.clip(
            1.0 + coefficients["clahe_slope"] * contrast_def
            * (1.0 - 0.5 * high_frequency),
            1.0,
            1.8,
        )
    )

    return {
        "red_alpha": red_alpha,
        "red_reference": "green",
        "shades_max_gain": 1.5,
        "shades_stretch": False,
        "udcp_omega": udcp_omega,
        "udcp_t0": udcp_t0,
        "clahe_clip_limit": clahe_clip_limit,
    }


def enhance_adaptive(image, return_details=False, controller=None):
    """
    Adaptive enhancement pipeline.

    Returns:
        enhanced image

    With return_details=True:
        enhanced image, degradation indicators and parameters
    """

    degradation = estimate_degradation(image)

    params = select_adaptive_parameters(degradation, controller=controller)

    # --------------------------------------------------
    # STEP 1: RED COMPENSATION
    # --------------------------------------------------

    red_corrected = correct(
        image,
        "red_comp",
        alpha=params["red_alpha"],
        reference=params["red_reference"],
    )

    # --------------------------------------------------
    # STEP 2: SHADES OF GRAY
    # --------------------------------------------------

    color_balanced = correct(
        red_corrected,
        "shades_of_gray",
        power=6.0,
        max_gain=params["shades_max_gain"],
        low=1.0,
        high=99.0,
        stretch=params["shades_stretch"],
    )

    # --------------------------------------------------
    # STEP 3: UDCP DEHAZING
    # --------------------------------------------------

    dehazed = dehaze(
        color_balanced,
        method="udcp",
        patch=15,
        omega=params["udcp_omega"],
        t0=params["udcp_t0"],
        refine=True,
    )

    # --------------------------------------------------
    # STEP 4: CLAHE CONTRAST ENHANCEMENT
    # --------------------------------------------------

    enhanced = enhance_contrast(
        dehazed,
        method="clahe",
        clip_limit=params["clahe_clip_limit"],
        tile_size=8,
    )

    if return_details:
        return enhanced, {
            "degradation": degradation,
            "parameters": params,
        }

    return enhanced
