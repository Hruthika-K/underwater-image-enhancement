"""Conservative scene-adaptive underwater enhancement.

This experimental pipeline is intentionally separate from ``fusion.py``.
It restores foreground color and lightness while keeping detected background
water close to the source image.
"""
import cv2
import numpy as np

from dip.dehazing import dehaze


def _scene_type(image):
    means = image.astype(np.float32).reshape(-1, 3).mean(axis=0)
    blue, green, red = means
    blue_excess = blue - max(green, red)
    green_excess = green - max(blue, red)
    if max(blue_excess, green_excess) < 12.0:
        return "mixed"
    return "blue" if blue_excess > green_excess else "green"


def _preserve_luminance(balanced, source):
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = float(cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[..., 0].mean())
    current = max(float(lab[..., 0].mean()), 1e-3)
    lab[..., 0] = np.clip(lab[..., 0] * target / current, 0.0, 255.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _haze_offset_and_stretch(image):
    values = image.astype(np.float32)
    low = np.minimum(
        np.percentile(values.reshape(-1, 3), 1.0, axis=0),
        60.0,
    )
    values = np.clip(values - low, 0.0, None)
    high = float(np.percentile(values.max(axis=2), 99.5))
    values = values / max(high, 1.0) * 239.0 + 6.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _global_color_precondition(image, scene):
    normalized = image.astype(np.float32) / 255.0
    blue, green, red = cv2.split(normalized)
    means = np.mean(normalized, axis=(0, 1))
    red_delta = means[1] - means[2]
    red = np.clip(
        red
        + (0.6 if scene == "blue" else 1.0)
        * red_delta
        * (1.0 - red)
        * green,
        0.0,
        1.0,
    )
    if means[0] < 0.7 * means[1]:
        blue = np.clip(
            blue + (means[1] - means[0]) * (1.0 - blue) * green,
            0.0,
            1.0,
        )
    corrected = cv2.merge([blue, green, red])
    p = 6.0
    illumination = np.power(
        np.mean(np.power(corrected, p), axis=(0, 1)),
        1.0 / p,
    )
    gains = np.clip(
        np.mean(illumination) / (illumination + 1e-6),
        0.6,
        2.0,
    )
    corrected = np.clip(corrected * gains, 0.0, 1.0)
    balanced = (corrected * 255.0).round().astype(np.uint8)
    balanced = _preserve_luminance(balanced, image)
    return _haze_offset_and_stretch(balanced)


def _normalize_lightness(image, target=132.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32)
    mean = float(np.mean(lightness))
    if mean < 1.0:
        return image.copy()
    if abs(mean - target) < 1.0:
        return image.copy()
    gamma = np.clip(
        np.log(target / 255.0) / np.log(mean / 255.0),
        0.82,
        1.18,
    )
    adjusted = np.power(lightness / 255.0, gamma) * 255.0
    lab[..., 0] = np.clip(adjusted, 0.0, 255.0).round().astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _conditional_chroma(image, threshold=0.16, strength=0.08):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt(
        (lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2
    )
    mean_chroma = float(np.mean(chroma)) / 128.0
    amount = float(np.clip((threshold - mean_chroma) / threshold, 0.0, 1.0))
    lab[..., 1:] = 128.0 + (lab[..., 1:] - 128.0) * (
        1.0 + min(0.15, strength * amount)
    )
    return cv2.cvtColor(
        np.clip(lab, 0.0, 255.0).round().astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _background_blue_bias(image, scene, strength=0.04):
    """Keep the background water cool without changing foreground colors."""
    scene_factor = {"blue": 1.0, "mixed": 0.7, "green": 0.4}[scene]
    normalized = image.astype(np.float32) / 255.0
    normalized[..., 0] = np.clip(
        normalized[..., 0] * (
            1.0 + scene_factor * float(np.clip(strength, 0.0, 0.25))
        ),
        0.0,
        1.0,
    )
    return (normalized * 255.0).round().astype(np.uint8)


def _water_probability(image):
    """Estimate background-water likelihood, not object degradation."""
    normalized = image.astype(np.float32) / 255.0
    blue, green, red = cv2.split(normalized)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    texture_level = max(float(np.percentile(texture, 70)), 1e-3)
    smooth_water = np.clip(1.0 - texture / texture_level, 0.0, 1.0)
    blue_or_green = np.clip(
        (np.maximum(blue, green) - red - 0.06) / 0.20,
        0.0,
        1.0,
    )
    return cv2.GaussianBlur(
        smooth_water * blue_or_green,
        (0, 0),
        4.0,
    )


def _foreground_weight(image, transmission, variant="no_transmission"):
    valid_variants = (
        "current",
        "no_transmission",
        "reduced_texture",
        "water_weak",
        "water_medium",
        "water_strong",
    )
    if variant not in valid_variants:
        raise ValueError(
            f"variant must be one of {valid_variants}"
        )
    normalized = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    if variant in ("no_transmission", "current"):
        weight = _selective_object_weight(image)
    else:
        texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        texture_scale = max(float(np.percentile(texture, 75)), 1e-3)
        texture_weight = np.clip(texture / texture_scale, 0.0, 1.0)
        if variant == "reduced_texture":
            texture_weight = 0.35 + 0.65 * texture_weight
        weight = texture_weight
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(
        np.float32
    ) / 255.0
    highlight = 1.0 - np.clip((lightness - 0.80) / 0.12, 0.0, 1.0)
    shadow = np.clip((lightness - 0.08) / 0.18, 0.0, 1.0)
    weight = weight * highlight * shadow
    if variant == "current":
        visible = np.clip((transmission - 0.25) / 0.45, 0.0, 1.0)
        weight *= visible
    water_strengths = {
        "water_weak": 0.25,
        "water_medium": 0.50,
        "water_strong": 0.75,
    }
    if variant in water_strengths:
        weight *= 1.0 - water_strengths[variant] * _water_probability(image)
    feathered = cv2.GaussianBlur(
        np.clip(weight, 0.0, 1.0),
        (0, 0),
        2.0,
    )
    feathered[feathered < 0.02] = 0.0
    return feathered


def _selective_object_weight(image):
    """Build a sparse objectness map from edges and local appearance changes."""
    normalized = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    local_detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3.0))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    chroma_detail = np.linalg.norm(
        lab[..., 1:] - cv2.GaussianBlur(lab[..., 1:], (0, 0), 3.0),
        axis=2,
    )
    edge_scale = max(float(np.percentile(edges, 85)), 1e-4)
    detail_scale = max(float(np.percentile(local_detail, 85)), 1e-4)
    chroma_scale = max(float(np.percentile(chroma_detail, 85)), 1e-4)
    evidence = (
        0.50 * np.clip(edges / edge_scale, 0.0, 1.0)
        + 0.30 * np.clip(local_detail / detail_scale, 0.0, 1.0)
        + 0.20 * np.clip(chroma_detail / chroma_scale, 0.0, 1.0)
    )
    threshold = float(np.percentile(evidence, 72))
    soft_evidence = np.clip(
        (evidence - threshold) / max(1.0 - threshold, 1e-6),
        0.0,
        1.0,
    )
    seed = (soft_evidence >= 0.35).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    seed = cv2.dilate(seed, kernel, iterations=1)
    weight = np.maximum(soft_evidence * 0.75, seed.astype(np.float32) * 0.35)
    return cv2.GaussianBlur(weight, (0, 0), 3.5)


def _foreground_stage(candidate, source, weight):
    alpha = np.clip(weight, 0.0, 1.0)[..., None]
    return np.clip(
        candidate.astype(np.float32) * alpha
        + source.astype(np.float32) * (1.0 - alpha),
        0.0,
        255.0,
    ).round().astype(np.uint8)


def _foreground_context(image, weight):
    """Keep local context for estimation without enhancing the water."""
    blurred = cv2.GaussianBlur(image, (0, 0), 12.0)
    context = np.clip(weight * 1.8, 0.0, 1.0)[..., None]
    return np.clip(
        image.astype(np.float32) * context
        + blurred.astype(np.float32) * (1.0 - context),
        0.0,
        255.0,
    ).round().astype(np.uint8)


def _foreground_gains(image, weight, scene):
    normalized = image.astype(np.float32) / 255.0
    selected = normalized[weight > 0.35]
    if selected.shape[0] < 64:
        return np.ones(3, dtype=np.float32)
    means = selected.mean(axis=0)
    target = float(np.mean(means))
    gains = np.clip(target / (means + 1e-6), 0.75, 1.8)
    if scene == "blue":
        gains[0] = min(gains[0], 1.03)
    elif scene == "mixed":
        gains = 1.0 + 0.5 * (gains - 1.0)
    gains[2] = min(gains[2], 1.8)
    return gains


def _foreground_balance(image, weight, scene, gains=None):
    normalized = image.astype(np.float32) / 255.0
    if gains is None:
        gains = _foreground_gains(image, weight, scene)
    applied = gains
    return (np.clip(normalized * applied, 0.0, 1.0) * 255).round().astype(
        np.uint8
    )


def _foreground_color_recovery(
    image,
    weight,
    scene,
    strength=0.55,
    gains=None,
):
    """Reduce foreground color cast while leaving final masking to compositing."""
    normalized = image.astype(np.float32) / 255.0
    if gains is None:
        gains = _foreground_gains(image, weight, scene)
    strength = float(np.clip(strength, 0.0, 1.0))
    damped_gains = 1.0 + strength * (gains - 1.0)
    corrected = np.clip(normalized * damped_gains, 0.0, 1.0)
    lab = cv2.cvtColor(
        (corrected * 255.0).round().astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    chroma_strength = 1.0 + 0.12 * strength
    lab[..., 1:] = 128.0 + (lab[..., 1:] - 128.0) * chroma_strength
    return cv2.cvtColor(
        np.clip(lab, 0.0, 255.0).round().astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _selective_color_recovery(image, weight, scene, strength=0.65):
    """Correct only the dominant foreground cast, with scene-specific limits."""
    normalized = image.astype(np.float32) / 255.0
    selected = normalized[weight > 0.35]
    if selected.shape[0] < 64:
        return image.copy()
    means = selected.mean(axis=0)
    blue, green, red = means
    gains = np.ones(3, dtype=np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    if scene == "green":
        excess = np.clip((green - max(blue, red) - 0.02) / 0.22, 0.0, 1.0)
        gains[1] -= 0.28 * strength * excess
        gains[2] += 0.10 * strength * excess
    elif scene == "blue":
        excess = np.clip((blue - max(green, red) - 0.02) / 0.22, 0.0, 1.0)
        gains[0] -= 0.24 * strength * excess
        gains[2] += 0.08 * strength * excess
    else:
        cool_excess = np.clip(
            (blue - max(green, red) - 0.03) / 0.25,
            0.0,
            1.0,
        )
        warm_deficit = np.clip(
            (0.90 * max(green, blue) - red) / 0.35,
            0.0,
            1.0,
        )
        gains[0] -= 0.16 * strength * cool_excess
        gains[1] += 0.06 * strength * warm_deficit
        gains[2] += 0.10 * strength * warm_deficit
    corrected = np.clip(normalized * gains, 0.0, 1.0)
    lab = cv2.cvtColor(
        (corrected * 255.0).round().astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    lab[..., 1:] = 128.0 + (lab[..., 1:] - 128.0) * (1.0 + 0.10 * strength)
    return cv2.cvtColor(
        np.clip(lab, 0.0, 255.0).round().astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _adaptive_color_recovery(
    image,
    weight,
    attenuation=True,
    red_recovery=True,
    strength=0.75,
):
    """Scale foreground cast correction with the measured relative excess."""
    normalized = image.astype(np.float32) / 255.0
    selected = normalized[weight > 0.35]
    if selected.shape[0] < 64:
        return image.copy()
    means = selected.mean(axis=0)
    blue, green, red = means
    chroma_mean = float(np.mean(means))
    strength = float(np.clip(strength, 0.0, 1.0))
    gains = np.ones(3, dtype=np.float32)
    if attenuation:
        blue_reference = max(green, red, 1e-3)
        green_reference = max(blue, red, 1e-3)
        blue_excess = np.clip(
            blue / blue_reference - 1.0,
            0.0,
            2.0,
        )
        green_excess = np.clip(
            green / green_reference - 1.0,
            0.0,
            2.0,
        )
        gains[0] -= 0.32 * strength * blue_excess
        gains[1] -= 0.32 * strength * green_excess
    if red_recovery:
        red_deficit = np.clip(
            (max(blue, green) - red - 0.04) / 0.35,
            0.0,
            1.0,
        )
        red_headroom = np.clip(1.0 - red, 0.0, 1.0)
        gains[2] += 0.18 * strength * red_deficit * red_headroom
    local = cv2.GaussianBlur(normalized, (0, 0), 5.0)
    local_cast = local - local.mean(axis=2, keepdims=True)
    global_cast = means - chroma_mean
    cast_alignment = np.clip(
        1.0 + 1.5 * np.sum(
            local_cast * global_cast[None, None, :],
            axis=2,
        ),
        0.65,
        1.35,
    )
    local_gains = 1.0 + (gains - 1.0) * cast_alignment[..., None]
    lightness = cv2.cvtColor(
        (normalized * 255.0).round().astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    )[..., 0].astype(np.float32) / 255.0
    local_gains[..., 2] = 1.0 + (
        local_gains[..., 2] - 1.0
    ) * np.sqrt(np.clip(1.0 - lightness, 0.0, 1.0))
    local_gains = np.clip(local_gains, 0.6, 2.0)
    corrected = np.clip(normalized * local_gains, 0.0, 1.0)
    lab = cv2.cvtColor(
        (corrected * 255.0).round().astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    return cv2.cvtColor(
        np.clip(lab, 0.0, 255.0).round().astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _restore_red(image, source, weight, scene):
    normalized = image.astype(np.float32) / 255.0
    source_float = source.astype(np.float32) / 255.0
    _, green, red = cv2.split(normalized)
    selected = source_float[weight > 0.35]
    if selected.shape[0] < 64:
        return image.copy()
    source_means = selected.mean(axis=0)
    red_ratio = source_means[2] / (max(source_means[0], source_means[1]) + 1e-6)
    deficit = np.clip((0.62 - red_ratio) / 0.62, 0.0, 1.0)
    if scene == "blue":
        amount = 0.10 * deficit
    elif scene == "green":
        amount = 0.16 * deficit
    else:
        amount = 0.08 * deficit
    gap = np.clip(
        cv2.GaussianBlur(green, (0, 0), 2.0)
        - cv2.GaussianBlur(red, (0, 0), 2.0),
        0.0,
        0.20,
    )
    correction = amount * gap * (1.0 - red)
    normalized[..., 2] = np.clip(red + correction, 0.0, 1.0)
    return (normalized * 255).round().astype(np.uint8)


def _adaptive_lightness(
    image,
    source,
    weight,
    scene,
    brightness_strength=1.7,
    contrast_strength=1.0,
    selective_color=False,
):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0]
    source_lightness = source_lab[..., 0]
    mean = float(source_lightness.mean()) / 255.0
    if mean < 0.42:
        gamma = 0.88
    elif mean < 0.55:
        gamma = 0.94
    else:
        gamma = 1.0
    brightness_strength = float(np.clip(brightness_strength, 0.0, 2.0))
    gamma = 1.0 + (gamma - 1.0) * brightness_strength
    # Values below one brighten when gamma is below one.
    adjusted = np.power(
        lightness.astype(np.float32) / 255.0,
        gamma,
    ) * 255.0
    clahe = cv2.createCLAHE(
        clipLimit=1.6 if scene == "mixed" else 1.8,
        tileGridSize=(8, 8),
    )
    local = clahe.apply(lightness)
    contrast_amount = (
        0.20 + 0.20 * float(np.mean(weight))
    ) * float(np.clip(contrast_strength, 0.0, 2.0))
    contrast_amount = min(contrast_amount, 0.75)
    enhanced = (
        adjusted * (1.0 - contrast_amount)
        + local.astype(np.float32) * contrast_amount
    )
    highlight = 1.0 - np.clip((source_lightness.astype(np.float32) / 255.0 - 0.80) / 0.12, 0.0, 1.0)
    enhanced = source_lightness.astype(np.float32) * (1.0 - highlight) + enhanced * highlight
    lab[..., 0] = np.clip(enhanced, 0.0, 255.0).round().astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _retinex_lightness(image, strength=0.30):
    """Recover uneven illumination on LAB lightness without changing hue."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32) / 255.0
    log_lightness = np.log(lightness + 1e-3)
    retinex = np.zeros_like(lightness)
    for sigma in (15.0, 80.0, 250.0):
        blurred = cv2.GaussianBlur(lightness, (0, 0), sigma)
        retinex += log_lightness - np.log(blurred + 1e-3)
    retinex /= 3.0
    low, high = np.percentile(retinex, [1.0, 99.0])
    normalized = np.clip(
        (retinex - low) / max(float(high - low), 1e-6),
        0.0,
        1.0,
    )
    enhanced = (
        lightness * (1.0 - strength)
        + normalized * strength
    )
    lab[..., 0] = np.clip(enhanced * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _adaptive_gamma_lightness(image, strength=1.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32) / 255.0
    mean = float(lightness.mean())
    gamma = 0.84 if mean < 0.42 else 0.92 if mean < 0.55 else 1.0
    strength = float(np.clip(strength, 0.0, 2.0))
    gamma = 1.0 + (gamma - 1.0) * strength
    lab[..., 0] = np.clip(
        np.power(lightness, gamma) * 255.0,
        0.0,
        255.0,
    ).round().astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _contrast_stage(
    image,
    source,
    weight,
    scene,
    method,
    brightness_strength=1.0,
    contrast_strength=1.0,
):
    if method == "clahe":
        return _adaptive_lightness(
            image,
            source,
            weight,
            scene,
            brightness_strength,
            contrast_strength,
        )
    if method == "retinex":
        return _adaptive_lightness(
            _retinex_lightness(image),
            source,
            weight,
            scene,
            brightness_strength,
            contrast_strength,
        )
    if method == "adaptive_gamma":
        return _adaptive_lightness(
            _adaptive_gamma_lightness(image, brightness_strength),
            source,
            weight,
            scene,
            brightness_strength,
            contrast_strength,
        )
    if method == "multiscale":
        retinex = _retinex_lightness(image, strength=0.24)
        gamma = _adaptive_gamma_lightness(image, brightness_strength)
        clahe = _adaptive_lightness(
            image,
            source,
            weight,
            scene,
            brightness_strength,
            contrast_strength,
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        ret_l = cv2.cvtColor(retinex, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        gamma_l = cv2.cvtColor(gamma, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        clahe_l = cv2.cvtColor(clahe, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        lab[..., 0] = (
            0.35 * ret_l + 0.30 * gamma_l + 0.35 * clahe_l
        ).round().astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    raise ValueError(
        "method must be 'clahe', 'retinex', 'adaptive_gamma', or 'multiscale'"
    )


def _enhance_whole_image(
    image,
    scene,
    method,
    contrast_method,
    dehaze_strength=0.6,
    color_strength=0.35,
    enable_red=False,
    brightness_strength=1.8,
    contrast_strength=1.2,
):
    """Run a restrained enhancement sequence for the background candidate."""
    full_weight = np.ones(image.shape[:2], dtype=np.float32)
    smoothed = cv2.bilateralFilter(image, 5, 20.0, 20.0)
    dehazed, transmission = dehaze(
        smoothed,
        method,
        t0=0.35,
        omega=0.82,
        return_t=True,
    )
    haze = float(np.clip(1.0 - np.percentile(transmission, 40), 0.0, 1.0))
    amount = float(np.clip(
        (0.30 + 0.35 * haze) * dehaze_strength,
        0.0,
        0.90,
    ))
    working = cv2.addWeighted(
        smoothed,
        1.0 - amount,
        dehazed,
        amount,
        0,
    )
    balanced = _adaptive_color_recovery(
        working,
        full_weight,
        attenuation=True,
        red_recovery=False,
        strength=color_strength,
    )
    balanced = _background_blue_bias(balanced, scene)
    restored = (
        _restore_red(balanced, image, full_weight, scene)
        if enable_red
        else balanced
    )
    denoised = cv2.bilateralFilter(restored, 5, 12.0, 12.0)
    method_name = "clahe" if contrast_method == "object_clahe" else contrast_method
    contrasted = _contrast_stage(
        denoised,
        image,
        full_weight,
        scene,
        method_name,
        brightness_strength=brightness_strength,
        contrast_strength=contrast_strength,
    )
    contrasted = _conditional_chroma(contrasted)
    return {
        "dehazed": dehazed,
        "dehaze_blend": working,
        "balanced": balanced,
        "red_restored": restored,
        "denoised": denoised,
        "illumination_contrast": contrasted,
        "output": contrasted,
        "transmission": transmission,
        "dehaze_amount": amount,
    }


def enhance_scene_adaptive(
    image,
    return_stages=False,
    contrast_method="clahe",
    mask_variant="no_transmission",
    enable_balance=True,
    balance_gains=None,
    enable_red=True,
    enable_contrast=True,
    color_strength=1.0,
    dehaze_strength=1.5,
    brightness_strength=2.0,
    contrast_strength=1.7,
    selective_color=False,
    adaptive_color=True,
    adaptive_attenuation=True,
    adaptive_red=True,
    composite_strength=3.0,
    include_whole_image=False,
    include_background_image=True,
):
    """Enhance one BGR uint8 image using conservative scene-aware processing.

    Enhancement candidates are composited once through the feathered
    foreground mask; the source image remains the background base.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 BGR image with shape (height, width, 3)")

    source_image = image
    scene = _scene_type(source_image)
    image = _normalize_lightness(
        _global_color_precondition(source_image, scene)
    )
    dehaze_method = (
        "udcp" if scene == "blue" else "red" if scene == "green" else "udcp"
    )
    weight = _foreground_weight(
        image,
        np.ones(image.shape[:2], dtype=np.float32),
        variant=mask_variant,
    )
    transmission_input = cv2.bilateralFilter(
        _foreground_context(image, weight),
        5,
        20.0,
        20.0,
    )
    dehazed, transmission = dehaze(
        transmission_input,
        dehaze_method,
        t0=0.35,
        omega=0.82,
        return_t=True,
    )
    haze = float(np.clip(1.0 - np.percentile(transmission, 40), 0.0, 1.0))
    dehaze_amount = float(np.clip(
        (0.30 + 0.35 * haze) * np.clip(dehaze_strength, 0.0, 2.0),
        0.0,
        0.90,
    ))
    dehazed_candidate = cv2.addWeighted(
        transmission_input,
        1.0 - dehaze_amount,
        dehazed,
        dehaze_amount,
        0,
    )
    object_dehaze = _foreground_stage(dehazed_candidate, image, weight)
    working = object_dehaze
    if enable_balance:
        if adaptive_color:
            balanced_candidate = _adaptive_color_recovery(
                working,
                weight,
                attenuation=adaptive_attenuation,
                red_recovery=adaptive_red,
                strength=color_strength,
            )
            balanced = _foreground_stage(
                balanced_candidate,
                working,
                weight,
            )
        elif selective_color:
            balanced_candidate = _selective_color_recovery(
                working,
                weight,
                scene,
                strength=color_strength,
            )
            balanced = _foreground_stage(
                balanced_candidate,
                working,
                weight,
            )
        else:
            balanced_candidate = _foreground_color_recovery(
                working,
                weight,
                scene,
                strength=color_strength,
                gains=balance_gains,
            )
            balanced = _foreground_stage(balanced_candidate, working, weight)
    else:
        balanced = working
    restored = (
        _foreground_stage(
            _restore_red(balanced, image, weight, scene),
            balanced,
            weight,
        )
        if enable_red
        else balanced
    )
    denoised = _foreground_stage(
        cv2.bilateralFilter(restored, 5, 12.0, 12.0),
        restored,
        weight,
    )
    retinex = _retinex_lightness(denoised)
    object_contrast = denoised
    if enable_contrast:
        method = "clahe" if contrast_method == "object_clahe" else contrast_method
        contrast_candidate = _contrast_stage(
            denoised,
            image,
            weight,
            scene,
            method,
            brightness_strength,
            contrast_strength,
        )
        object_contrast = _foreground_stage(
            contrast_candidate,
            denoised,
            weight,
        )
        object_contrast = _conditional_chroma(object_contrast)
    enhanced_foreground = object_contrast
    alpha = np.clip(weight, 0.0, 1.0)
    composite_strength = float(np.clip(composite_strength, 0.0, 3.0))
    alpha = 1.0 - np.power(1.0 - alpha, composite_strength)
    confident = np.clip((weight - 0.20) / 0.25, 0.0, 1.0)
    confident = confident * confident * (3.0 - 2.0 * confident)
    alpha = alpha + confident * (0.95 - alpha)
    alpha = alpha[..., None].astype(np.float32)
    final = np.clip(
        enhanced_foreground.astype(np.float32) * alpha
        + image.astype(np.float32) * (1.0 - alpha),
        0.0,
        255.0,
    ).round().astype(np.uint8)
    foreground_only = np.clip(
        enhanced_foreground.astype(np.float32) * alpha,
        0.0,
        255.0,
    ).round().astype(np.uint8)
    whole = None
    result = final
    if include_whole_image or include_background_image:
        whole = _enhance_whole_image(
            final,
            scene,
            dehaze_method,
            contrast_method,
        )
        if include_whole_image:
            result = whole["output"]
        else:
            background_alpha = (1.0 - alpha).astype(np.float32)
            result = np.clip(
                whole["output"].astype(np.float32) * background_alpha
                + final.astype(np.float32) * (1.0 - background_alpha),
                0.0,
                255.0,
            ).round().astype(np.uint8)
    if not return_stages:
        return result
    return {
        "input": image,
        "denoised_input": transmission_input,
        "dehazed": dehazed,
        "transmission": transmission,
        "dehaze_amount": dehaze_amount,
        "dehaze_blend": working,
        "object_dehaze": object_dehaze,
        "foreground_weight": weight,
        "mask_variant": mask_variant,
        "balanced": balanced,
        "balance_gains": (
            _foreground_gains(working, weight, scene)
            if enable_balance and balance_gains is None
            else balance_gains
        ),
        "color_strength": color_strength,
        "red_restored": restored,
        "denoised": denoised,
        "retinex": retinex,
        "contrast_method": contrast_method,
        "illumination_contrast": object_contrast,
        "enhanced_foreground": foreground_only,
        "foreground_output": final,
        "whole_image_stages": whole,
        "output": result,
    }