"""Multi-scale underwater enhancement and pyramid fusion.

The pipeline uses underwater-aware white balance, adaptive dehazing, Lab
lightness enhancement, detail-aware weights, and normalized pyramid fusion.
Public image inputs and outputs are uint8 BGR images.
"""
import cv2
import numpy as np

from dip.color_correction import correct
from dip.contrast import enhance_contrast
from dip.dehazing import dehaze


def gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(img, levels):
    gaussian = gaussian_pyramid(img, levels)
    laplacian = []
    for i in range(levels - 1):
        up = cv2.pyrUp(
            gaussian[i + 1],
            dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]),
        )
        laplacian.append(gaussian[i] - up)
    laplacian.append(gaussian[-1])
    return laplacian


def reconstruct(laplacian):
    image = laplacian[-1]
    for layer in reversed(laplacian[:-1]):
        image = cv2.pyrUp(
            image, dstsize=(layer.shape[1], layer.shape[0])
        ) + layer
    return image


def _normalize(value, low_percentile=2.0, high_percentile=98.0):
    """Normalize a map robustly so isolated extrema do not dominate weights."""
    value = np.asarray(value, dtype=np.float32)
    low, high = np.percentile(value, [low_percentile, high_percentile])
    if high - low <= 1e-6:
        return np.zeros_like(value)
    return np.clip((value - low) / (high - low), 0.0, 1.0)


def weight_maps(img):
    """Return the unnormalized Laplacian + saliency + saturation map."""
    image = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = _normalize(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    saliency = np.linalg.norm(
        cv2.GaussianBlur(lab, (5, 5), 0) - mean_lab, axis=2
    )
    saliency = _normalize(saliency)

    saturation = _normalize(np.std(image, axis=2))
    cyan = np.clip((image[..., 0] + image[..., 1]) / 2.0 - image[..., 2], 0.0, 1.0)
    saturation *= 1.0 - 0.35 * _normalize(cyan)
    return laplacian + saliency + saturation


def adaptive_gamma(image, target=0.48, min_gamma=0.5, max_gamma=1.5):
    """Find a bounded power-law gamma that moves mean luminance toward target."""
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    mean_luminance = float(gray.mean())
    if mean_luminance < 1e-6:
        return 1.0
    low, high = min_gamma, max_gamma
    for _ in range(24):
        middle = (low + high) / 2
        adjusted_mean = float(np.power(gray, middle).mean())
        if adjusted_mean > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _adaptive_preconditioning(image):
    """Lift compressed shadows and suppress dominant blue/green casts."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32) / 255.0
    mean_lightness = float(lightness.mean())
    contrast = float(np.percentile(lightness, 95) - np.percentile(lightness, 5))

    strength = float(np.clip((0.50 - contrast) / 0.50, 0.0, 0.35))
    if mean_lightness < 0.35:
        strength = max(strength, float(np.clip((0.35 - mean_lightness) * 1.5, 0.0, 0.35)))
    if strength > 0:
        enhanced = enhance_contrast(image, "clahe", clip_limit=1.8)
        image = cv2.addWeighted(image, 1.0 - strength, enhanced, strength, 0)

    channels = image.astype(np.float32) / 255.0
    blue_mean, green_mean, red_mean = channels.reshape(-1, 3).mean(axis=0)
    blue_excess = blue_mean - max(green_mean, red_mean)
    if blue_excess > 0.06 and red_mean < green_mean * 0.9:
        suppression = float(np.clip((blue_excess - 0.06) * 0.8, 0.0, 0.12))
        channels[..., 0] *= 1.0 - suppression

    dominant_channel = int(np.argmax([blue_mean, green_mean, red_mean]))
    if dominant_channel in (0, 1):
        means = [blue_mean, green_mean, red_mean]
        other_mean = (means[(dominant_channel + 1) % 3] + means[(dominant_channel + 2) % 3]) / 2.0
        dominance = means[dominant_channel] / (other_mean + 1e-6)
        if dominance > 1.15:
            suppression = float(np.clip((dominance - 1.15) * 0.35, 0.0, 0.22))
            channels[..., dominant_channel] *= 1.0 - suppression

    image = (np.clip(channels, 0.0, 1.0) * 255).round().astype(np.uint8)
    return image


def _adaptive_dehaze_strength(image, requested):
    """Increase DCP contribution only for visibly flat, hazy images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    percentile_contrast = float(
        np.percentile(gray, 95) - np.percentile(gray, 5)
    )
    haze_boost = float(np.clip((0.50 - percentile_contrast) / 0.50, 0.0, 1.0))
    channels = image.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0
    blue_mean, green_mean, red_mean = channels
    blue_cast = float(np.clip(
        (blue_mean - max(green_mean, red_mean) - 0.04) / 0.20,
        0.0,
        1.0,
    ))
    return float(np.clip(
        0.75 * requested + 0.12 * haze_boost + 0.08 * blue_cast,
        0.0,
        0.8,
    ))


def _haze_score(image):
    """Estimate flatness/cast haze on a stable [0, 1] scale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    flatness = np.clip((120.0 - laplacian) / 120.0, 0.0, 1.0)
    existing = float(np.clip(
        0.65 * (0.50 - contrast) / 0.50 + 0.35 * flatness,
        0.0,
        1.0,
    ))
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(
        np.float32
    ) / 255.0
    pedestal = float(np.clip(
        (np.percentile(lightness, 1) - 0.08) / 0.25,
        0.0,
        1.0,
    ))
    return max(existing, pedestal)


def _dehaze_keep_color(image, method, chroma_keep=0.25, t0=0.3):
    """Use dehazing for lightness while retaining most source chroma."""
    dehazed = cv2.cvtColor(
        dehaze(image, method, t0=t0),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    dehazed[..., 1:] = source[..., 1:] + chroma_keep * (
        dehazed[..., 1:] - source[..., 1:]
    )
    return cv2.cvtColor(
        np.clip(dehazed, 0.0, 255.0).round().astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _lin(value):
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def _srgb(value):
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * value ** (1.0 / 2.4) - 0.055,
    )


def _xyz_y(lightness):
    return np.where(
        lightness > 8.0,
        ((lightness + 16.0) / 116.0) ** 3,
        lightness / 903.3,
    )


def _hue_safe(before, after):
    """Take lightness from ``after`` while retaining ``before`` color."""
    before_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
    before_l = before_lab[..., 0].astype(np.float32) * 100.0 / 255.0
    after_l = after_lab[..., 0].astype(np.float32) * 100.0 / 255.0
    ratio = np.clip(
        (_xyz_y(after_l) + 1e-4) / (_xyz_y(before_l) + 1e-4),
        0.3,
        3.0,
    )[..., None]
    linear = _lin(before.astype(np.float32) / 255.0) * ratio
    return (_srgb(linear) * 255.0).round().clip(0, 255).astype(np.uint8)


def _luminance_stretch(image, haze, low=0.5, high=99.5, max_gain=1.8):
    """Remove the haze pedestal from lightness while preserving color ratios."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(lightness, [low, high])
    gain = float(np.clip(255.0 / max(float(hi - lo), 1.0), 1.0, max_gain))
    pivot = float(lightness.mean())
    stretched = np.clip(pivot + (lightness - pivot) * gain, 0.0, 255.0)
    highlight_protection = 1.0 - np.clip((lightness - 0.82) / 0.12, 0.0, 1.0)
    amount = haze * highlight_protection
    lab[..., 0] = (
        lightness * (1.0 - amount) + stretched * amount
    ).round().astype(np.uint8)
    return _hue_safe(image, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def _match_exposure(image, target=0.48, min_gamma=0.58):
    """Brighten low-light images on Lab lightness without darkening highlights."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32) / 255.0
    if float(lightness.mean()) >= target:
        return image
    low, high = min_gamma, 1.0
    for _ in range(20):
        middle = (low + high) / 2.0
        if float(np.power(lightness, middle).mean()) > target:
            low = middle
        else:
            high = middle
    gamma = (low + high) / 2.0
    adjusted = np.power(lightness, gamma).clip(0.0, 1.0)
    highlight_protection = 1.0 - np.clip((lightness - 0.82) / 0.12, 0.0, 1.0)
    adjusted = lightness * (1.0 - highlight_protection) + adjusted * highlight_protection
    lab[..., 0] = (adjusted * 255.0).round().astype(np.uint8)
    return _hue_safe(image, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def _cast_severity(image):
    """Measure how strongly blue or green dominates the source."""
    blue, green, red = image.astype(np.float32).reshape(-1, 3).mean(axis=0)
    return float(np.clip((max(blue, green) / (red + 1e-6) - 1.2), 0.0, 1.0))


def _scene_type(image):
    """Classify the dominant open-water cast as green or blue."""
    blue, green, _ = image.astype(np.float32).reshape(-1, 3).mean(axis=0)
    return "green" if green > blue else "blue"


def _foreground_weight(image):
    """Estimate textured subject regions while excluding water and highlights."""
    normalized = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    texture_level = float(np.percentile(texture, 70))
    texture_weight = np.clip(
        texture / max(texture_level, 1e-3),
        0.0,
        1.0,
    )
    blue, green, red = cv2.split(normalized)
    water_tint = np.clip(
        (np.maximum(blue, green) - red - 0.08) / 0.25,
        0.0,
        1.0,
    )
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(
        np.float32
    ) / 255.0
    highlight_protection = 1.0 - np.clip(
        (lightness - 0.78) / 0.14,
        0.0,
        1.0,
    )
    shadow_protection = np.clip((lightness - 0.10) / 0.20, 0.0, 1.0)
    foreground = texture_weight * (1.0 - 0.75 * water_tint)
    foreground *= highlight_protection * shadow_protection
    return cv2.GaussianBlur(
        np.clip(foreground, 0.0, 1.0),
        (0, 0),
        2.0,
    )


def _gain_white_balance(
    image,
    strength,
    cap=3.5,
    red_cap=3.5,
    foreground_weight=None,
):
    """Apply stronger mean-gain correction without boosting blue."""
    f = image.astype(np.float32) / 255.0
    if foreground_weight is None:
        selected = f.reshape(-1, 3)
    else:
        selected = f[foreground_weight > 0.35]
        if selected.shape[0] < 64:
            selected = f.reshape(-1, 3)
    means = selected.mean(axis=0)
    gains = np.clip(
        means.mean() / (means + 1e-6),
        1.0 / cap,
        cap,
    )
    gains[2] = min(
        means.mean() / (means[2] + 1e-6),
        red_cap,
    )
    gains = 1.0 + strength * (gains - 1.0)
    gains[0] = min(gains[0], 1.1)
    correction_strength = strength
    if foreground_weight is not None:
        correction_strength *= 0.25 + 0.75 * foreground_weight[..., None]
    applied_gains = 1.0 + correction_strength * (gains - 1.0)
    return (
        np.clip(f * applied_gains, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)


def _restore_warmth(output, source, strength=0.8):
    """Restore warmth in gated subject mid-tones, not water or highlights."""
    severity = _cast_severity(source)
    if severity <= 0.0:
        return output
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0] / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    gate = (
        np.clip((lightness - 0.15) / 0.20, 0.0, 1.0)
        * (1.0 - np.clip((lightness - 0.70) / 0.25, 0.0, 1.0))
    )
    gate *= 1.0 - np.clip((-b - 4.0) / 12.0, 0.0, 1.0)
    gate *= 1.0 - np.clip(a / 20.0, 0.0, 1.0)
    gate = cv2.GaussianBlur(gate, (0, 0), 6.0)
    if float(gate.sum()) < 64.0:
        return output

    weights = gate / (float(gate.sum()) + 1e-6)
    delta_a = float(np.clip(-(a * weights).sum(), 0.0, 18.0))
    delta_b = float(np.clip(-(b * weights).sum(), 0.0, 25.0))
    delta_a *= strength * severity
    delta_b *= strength * severity
    lab[..., 1] = np.clip(a + delta_a * gate + 128.0, 0.0, 255.0)
    lab[..., 2] = np.clip(b + delta_b * gate + 128.0, 0.0, 255.0)
    return cv2.cvtColor(lab.round().astype(np.uint8), cv2.COLOR_LAB2BGR)


def _limit_color(
    output,
    source,
    max_shift=8.0,
    max_chroma=45.0,
    denoise_sigma=1.0,
):
    """Keep chroma changes within a bounded budget relative to the source."""
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[..., 1:] - 128.0
    ab = cv2.GaussianBlur(ab, (0, 0), denoise_sigma)
    severity = _cast_severity(source)
    max_shift = 6.0 + 26.0 * severity

    shift = (
        ab.reshape(-1, 2).mean(axis=0)
        - (source_lab[..., 1:] - 128.0).reshape(-1, 2).mean(axis=0)
    )
    excess = np.sign(shift) * np.maximum(np.abs(shift) - max_shift, 0.0)
    ab -= excess

    source_ab = cv2.GaussianBlur(source_lab[..., 1:] - 128.0, (0, 0), 4.0)
    red_cap = source_ab[..., 0] + 22.0 * severity + 8.0 * (1.0 - severity)
    red_excess = np.maximum(ab[..., 0] - red_cap, 0.0)
    ab[..., 0] = np.minimum(ab[..., 0], red_cap) + 8.0 * np.tanh(
        red_excess / 8.0
    )

    source_chroma = np.linalg.norm(source_ab, axis=2)
    source_chroma = cv2.GaussianBlur(
        source_chroma,
        (0, 0),
        4.0,
    )[..., None]
    cap = np.maximum(max_chroma, 1.25 * source_chroma)
    chroma = np.linalg.norm(ab, axis=2, keepdims=True) + 1e-6
    knee = 0.8 * cap
    over = np.maximum(chroma - knee, 0.0)
    new_chroma = chroma - over + 0.2 * cap * np.tanh(
        over / np.maximum(0.2 * cap, 1e-6)
    )
    ab *= new_chroma / chroma
    lab[..., 1:] = np.clip(ab + 128.0, 0.0, 255.0)
    return cv2.cvtColor(lab.round().astype(np.uint8), cv2.COLOR_LAB2BGR)


def _protect_source_highlights(output, source, threshold=0.78):
    """Keep bright source regions from becoming white or chromatically shifted."""
    output_lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lightness = source_lab[..., 0] / 255.0
    protection = np.clip(
        (source_lightness - threshold) / (1.0 - threshold),
        0.0,
        1.0,
    )
    protection = cv2.GaussianBlur(protection, (0, 0), 2.0)
    output_lab = (
        output_lab * (1.0 - protection[..., None])
        + source_lab * protection[..., None]
    )
    return cv2.cvtColor(
        np.clip(output_lab, 0.0, 255.0).round().astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _water_weight(image):
    """Estimate smooth, strongly tinted water regions."""
    f = image.astype(np.float32) / 255.0
    blue, green, red = cv2.split(f)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    texture = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(gray, cv2.CV_32F)),
        (0, 0),
        6.0,
    )
    flat = 1.0 - np.clip(
        texture / (np.percentile(texture, 70) + 1e-6),
        0.0,
        1.0,
    )
    tint = np.clip(
        (np.maximum(blue, green) - red - 0.10) / 0.25,
        0.0,
        1.0,
    )
    return cv2.GaussianBlur(flat * tint, (0, 0), 8.0)


def _adaptive_dehaze_method(image):
    means = image.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0
    blue_mean, green_mean, red_mean = means
    if red_mean / (blue_mean + 1e-6) < 0.65:
        return "udcp"
    if blue_mean > green_mean * 1.12 and red_mean < green_mean * 0.85:
        return "red"
    return "dcp"


def _post_fusion_color_balance(image):
    """Preserve the historical hook without applying a second color shift."""
    return image


def _safe_white_balance(img, p=6, max_gain=1.6):
    """Apply capped shades-of-gray balancing without boosting blue."""
    image = img.astype(np.float32) / 255.0
    estimates = np.array([
        np.mean(image[..., channel] ** p) ** (1.0 / p)
        for channel in range(3)
    ])
    gains = estimates.mean() / (estimates + 1e-6)
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    gains[0] = min(gains[0], 1.1)
    return (np.clip(image * gains, 0.0, 1.0) * 255).round().astype(np.uint8)


def _masked_white_balance(img, mask, p=6, max_gain=1.6):
    """Estimate shades-of-gray gains from selected foreground pixels."""
    image = img.astype(np.float32) / 255.0
    selected = image[mask]
    if selected.shape[0] < 64:
        return _safe_white_balance(img, p=p, max_gain=max_gain)
    estimates = np.mean(selected ** p, axis=0) ** (1.0 / p)
    gains = estimates.mean() / (estimates + 1e-6)
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    gains[0] = min(gains[0], 1.1)
    return (np.clip(image * gains, 0.0, 1.0) * 255).round().astype(np.uint8)


def _safe_red_compensation(img, alpha=0.4):
    """Recover only a capped portion of the green-to-red gap."""
    image = img.astype(np.float32) / 255.0
    blue, green, red = cv2.split(image)
    gap = np.clip(
        cv2.GaussianBlur(green, (0, 0), 3.0)
        - cv2.GaussianBlur(red, (0, 0), 3.0),
        0.0,
        0.25,
    )
    valid_red = np.clip((red + 0.005) / 0.08, 0.0, 1.0)
    gray = cv2.cvtColor((image * 255.0).round().astype(np.uint8), cv2.COLOR_BGR2GRAY)
    texture = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    texture_low, texture_high = np.percentile(texture, [60, 90])
    texture_gate = np.clip(
        (texture - texture_low) / max(float(texture_high - texture_low), 1.0),
        0.0,
        1.0,
    )
    lightness = cv2.cvtColor(
        (image * 255.0).round().astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    )[..., 0].astype(np.float32) / 255.0
    texture_gate *= 1.0 - np.clip((lightness - 0.82) / 0.12, 0.0, 1.0)
    correction = alpha * gap * (1.0 - red) * green * valid_red * texture_gate
    correction *= 1.0 - np.clip((green - 0.6) / 0.3, 0.0, 1.0)
    correction *= 1.0 - 0.35 * np.clip((blue - green) / 0.25, 0.0, 1.0)
    image[..., 2] = np.clip(
        red + cv2.GaussianBlur(correction, (0, 0), 1.0),
        0.0,
        1.0,
    )
    return (image * 255).round().astype(np.uint8)


def adaptive_enhance(img):
    """Apply foreground-aware white balance, adaptive dehazing, and CLAHE."""
    haze = _haze_score(img)
    _, preliminary_t = dehaze(img, "dcp", t0=0.3, return_t=True)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    transmission_cutoff = np.percentile(preliminary_t, 60)
    texture_cutoff = np.percentile(texture, 45)
    foreground = (preliminary_t >= transmission_cutoff) & (texture >= texture_cutoff)
    balanced = _masked_white_balance(img, foreground, max_gain=1.25)
    method = _adaptive_dehaze_method(balanced)
    dehazed, transmission = dehaze(
        balanced,
        method,
        t0=0.3,
        return_t=True,
    )
    image = dehazed.astype(np.float32)
    _, green_mean, red_mean = image.reshape(-1, 3).mean(axis=0)
    red_deficit = float(np.clip(
        1.0 - red_mean / (green_mean + 1e-6),
        0.0,
        1.0,
    ))
    red_corrected = _safe_red_compensation(
        dehazed,
        alpha=0.4 + 0.15 * red_deficit,
    )
    clahe_result = enhance_contrast(red_corrected, "clahe", clip_limit=2.0)
    result = cv2.addWeighted(
        red_corrected,
        0.8,
        clahe_result,
        0.2,
        0,
    )
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    foreground_weight = cv2.GaussianBlur(
        foreground.astype(np.float32),
        (0, 0),
        9.0,
    )
    foreground_luminance = float(
        (gray * foreground_weight).sum() / (foreground_weight.sum() + 1e-6)
    )
    target = float(np.clip(0.42 - foreground_luminance, 0.0, 0.15))
    if target > 0:
        values = result.astype(np.float32) / 255.0
        highlight_mask = 1.0 - np.clip((gray - 0.85) / 0.15, 0.0, 1.0)
        lift_weight = foreground_weight * highlight_mask
        exponent = 1.0 - np.clip(target * 1.5, 0.0, 0.2) * lift_weight[..., None]
        values = np.power(values, exponent)
        water = cv2.GaussianBlur(
            (transmission < np.percentile(transmission, 35)).astype(np.float32),
            (0, 0),
            4.0,
        )
        highlight = np.clip((values - 0.8) / 0.2, 0.0, 1.0)
        rolled = values / (1.0 + 0.18 * values)
        values = values * (1.0 - 0.3 * water[..., None] * highlight) + rolled * (
            0.3 * water[..., None] * highlight
        )
        result = (np.clip(values, 0.0, 1.0) * 255).round().astype(np.uint8)
    result = _luminance_stretch(result, haze)
    result = _match_exposure(result)
    result = _restore_warmth(result, img)
    return _limit_color(result, img)


def _lean_enhance(
    img,
    dehaze_strength=0.7,
    clahe_clip=2.0,
    enable_white_balance=True,
    enable_red_restore=True,
    enable_contrast=True,
):
    """Run the minimal color, haze, contrast, and safety pipeline."""
    haze = _haze_score(img)
    scene = _scene_type(img)
    severity = _cast_severity(img)
    foreground_weight = _foreground_weight(img)
    if scene == "blue":
        chroma_keep, sog_gain = 0.0, 1.35
    else:
        chroma_keep, sog_gain = 0.5, 1.6 + severity
    method = _adaptive_dehaze_method(img)
    dehazed = _dehaze_keep_color(
        img,
        method,
        chroma_keep=chroma_keep,
        t0=0.35,
    )
    effective_strength = _adaptive_dehaze_strength(img, dehaze_strength)
    blended = cv2.addWeighted(
        img,
        1.0 - effective_strength,
        dehazed,
        effective_strength,
        0,
    )
    if not enable_white_balance:
        balanced = blended
    elif scene == "green":
        balanced = _gain_white_balance(
            blended,
            strength=0.8 + 0.2 * severity,
            foreground_weight=foreground_weight,
        )
    else:
        balanced = correct(
            blended,
            "shades_of_gray",
            max_gain=sog_gain,
            stretch=False,
        )
    balanced_lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB)
    source_lightness = source_lab[..., 0].astype(np.float32) / 255.0
    highlight_mask = 1.0 - np.clip((source_lightness - 0.78) / 0.14, 0.0, 1.0)
    balanced_lab = (
        source_lab.astype(np.float32) * (1.0 - highlight_mask[..., None])
        + balanced_lab.astype(np.float32) * highlight_mask[..., None]
    ).round().clip(0, 255).astype(np.uint8)
    balanced = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)
    if enable_red_restore:
        red_alpha = float(np.clip(
            (0.15 + 0.24 * severity) * 1.0,
            0.05,
            0.7,
        ))
        red_corrected = _safe_red_compensation(balanced, alpha=red_alpha)
        source_means = img.astype(np.float32).reshape(-1, 3).mean(axis=0)
        source_red_ratio = source_means[2] / (
            max(source_means[0], source_means[1]) + 1e-6
        )
        red_deficit = np.clip((0.7 - source_red_ratio) / 0.7, 0.0, 1.0)
        red_gain = 1.0 + 0.45 * severity * (
            0.20 + 0.35 * red_deficit
        )
        red_gain = min(red_gain, 1.35)
        if red_gain > 1.0:
            red_corrected = red_corrected.astype(np.float32)
            red_strength = 0.25 + 0.75 * foreground_weight
            red_strength *= 1.0 - np.clip(
                (cv2.cvtColor(red_corrected, cv2.COLOR_BGR2LAB)[..., 0].astype(
                    np.float32
                ) / 255.0 - 0.80) / 0.14,
                0.0,
                1.0,
            )
            pixel_gain = 1.0 + (red_gain - 1.0) * red_strength
            red_corrected[..., 2] = np.clip(
                red_corrected[..., 2] * pixel_gain,
                0.0,
                255.0,
            )
            red_corrected = red_corrected.round().astype(np.uint8)
    else:
        red_corrected = balanced
    contrast = enhance_contrast(
        red_corrected,
        "clahe",
        clip_limit=max(clahe_clip, 2.0),
    )
    contrast_strength = float(np.clip(0.15 + 0.55 * haze, 0.15, 0.6))
    base_lab = cv2.cvtColor(red_corrected, cv2.COLOR_BGR2LAB)
    contrast_lab = cv2.cvtColor(contrast, cv2.COLOR_BGR2LAB)
    base_l = base_lab[..., 0].astype(np.float32)
    highlight_protection = 1.0 - np.clip((base_l / 255.0 - 0.82) / 0.12, 0.0, 1.0)
    blend_amount = contrast_strength * highlight_protection
    blended_lab = base_lab.copy()
    if enable_contrast:
        blended_lab[..., 0] = (
            base_l * (1.0 - blend_amount)
            + contrast_lab[..., 0].astype(np.float32) * blend_amount
        ).round().astype(np.uint8)
    result = _hue_safe(
        red_corrected,
        cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR),
    )
    result = _luminance_stretch(result, haze)
    source_lightness = float(
        cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean() / 255.0
    )
    target_lightness = float(np.clip(
        max(0.50, 1.00 * source_lightness),
        0.50,
        0.62,
    ))
    result = _match_exposure(result, target=target_lightness)
    result = _protect_source_highlights(result, img)
    return _limit_color(result, img)


def fuse(
    img,
    levels=5,
    sharpen_sigma=2.0,
    sharpen_amount=0.35,
    clahe_clip=2.0,
    color_correct=True,
    return_parts=False,
    gamma=None,
    delta=1e-6,
    dehaze_method="dcp",
    dehaze_strength=0.7,
    contrast_strength=0.0,
    output_sharpen=0.2,
    pipeline="lean",
    enable_white_balance=True,
    enable_red_restore=True,
    enable_contrast=True,
):
    """Apply the requested enhancement and multi-scale fusion pipeline.

    ``pipeline="lean"`` uses the minimal default path. Set ``pipeline="fusion"``
    to use the multi-input Laplacian pyramid path.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("fuse expects a color image with shape (height, width, 3)")
    if gamma is not None and gamma <= 0:
        raise ValueError("gamma must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if sharpen_sigma < 0:
        raise ValueError("sharpen_sigma must be non-negative")
    if dehaze_method not in (None, "dcp", "udcp", "red"):
        raise ValueError("dehaze_method must be None, 'dcp', 'udcp', or 'red'")
    if not 0 <= dehaze_strength <= 1:
        raise ValueError("dehaze_strength must be between 0 and 1")
    if not 0 <= contrast_strength <= 1:
        raise ValueError("contrast_strength must be between 0 and 1")
    if not 0 <= output_sharpen <= 1:
        raise ValueError("output_sharpen must be between 0 and 1")
    if pipeline not in ("lean", "fusion"):
        raise ValueError("pipeline must be 'lean' or 'fusion'")
    if pipeline == "lean":
        output = _lean_enhance(
            img,
            dehaze_strength=dehaze_strength,
            clahe_clip=clahe_clip,
            enable_white_balance=enable_white_balance,
            enable_red_restore=enable_red_restore,
            enable_contrast=enable_contrast,
        )
        return (output, [], []) if return_parts else output

    working = img.copy()
    haze = _haze_score(working)
    if dehaze_method == "dcp":
        dehaze_method = _adaptive_dehaze_method(working)
    if dehaze_method is not None and dehaze_strength > 0:
        dehazed = _dehaze_keep_color(working, dehaze_method)
        effective_dehaze_strength = _adaptive_dehaze_strength(
            working, dehaze_strength
        )
        working = cv2.addWeighted(
            working,
            1.0 - effective_dehaze_strength,
            dehazed,
            effective_dehaze_strength,
            0,
        )

    if color_correct:
        working = correct(working, "shades_of_gray")
        means = working.astype(np.float32).reshape(-1, 3).mean(axis=0)
        b_mean, g_mean, r_mean = means
        base = working
        red_ratio = r_mean / (g_mean + 1e-6)
        blue_trigger = b_mean > g_mean * 1.2 and red_ratio < 0.75
        if red_ratio < 0.72 or blue_trigger:
            blue_adjustment = max(0.0, b_mean / (g_mean + 1e-6) - 1.2)
            alpha = float(np.clip(
                (0.72 - red_ratio) * 1.6 + blue_adjustment * 0.15,
                0.12,
                0.45,
            ))
            base = _safe_red_compensation(base, alpha=alpha)
        cast = max(b_mean, g_mean) / (r_mean + 1e-6)
        gray_world_strength = float(np.clip(
            0.4 + 0.25 * (cast - 1.5),
            0.4,
            0.65,
        ))
        base = correct(
            base,
            "gray_world",
            max_gain=2.5,
            strength=gray_world_strength,
        )
    else:
        base = working
    effective_contrast = max(contrast_strength, 0.5 * haze)
    if effective_contrast > 0:
        contrast_base = enhance_contrast(base, "clahe", clip_limit=2.0)
        base = cv2.addWeighted(
            base, 1.0 - effective_contrast,
            contrast_base, effective_contrast,
            0,
        )
    base = _adaptive_preconditioning(base)
    white_balanced = base.astype(np.float32) / 255.0

    effective_gamma = (
        adaptive_gamma(white_balanced) if gamma is None else gamma
    )
    i1 = np.power(white_balanced, effective_gamma)
    blurred = cv2.GaussianBlur(
        white_balanced, (0, 0), max(sharpen_sigma, 0.1)
    )
    detail = white_balanced - blurred
    i2 = np.clip(white_balanced + sharpen_amount * detail, 0.0, 1.0)
    lab = cv2.cvtColor((white_balanced * 255).round().astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.GaussianBlur(l, (3, 3), 0)
    enhanced_l = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l)
    clahe_input = cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2BGR)
    clahe_float = clahe_input.astype(np.float32) / 255.0
    broad_blur = cv2.GaussianBlur(
        white_balanced, (0, 0), max(sharpen_sigma * 2.0, 0.2)
    )
    local_contrast = np.clip(
        white_balanced + 0.5 * sharpen_amount * (white_balanced - broad_blur),
        0.0,
        1.0,
    )
    high_pass = local_contrast
    inputs = [i1, i2, clahe_float, high_pass]
    return_inputs = [
        (item * 255).round().clip(0, 255).astype(np.uint8) for item in inputs
    ]

    raw_weights = [
        weight_maps((item * 255).round().astype(np.uint8)) for item in inputs
    ]
    denominator = sum(raw_weights) + len(raw_weights) * delta
    weights = [(weight + delta) / denominator for weight in raw_weights]

    height, width = img.shape[:2]
    max_levels = max(1, int(np.log2(min(height, width))) - 2)
    levels = max(1, min(levels, max_levels))

    fused = None
    for image, weight in zip(inputs, weights):
        laplacian = laplacian_pyramid(image, levels)
        gaussian = gaussian_pyramid(weight.astype(np.float32), levels)
        part = [
            layer * scale[..., None]
            for layer, scale in zip(laplacian, gaussian)
        ]
        fused = part if fused is None else [
            first + second for first, second in zip(fused, part)
        ]

    output_float = np.clip(reconstruct(fused), 0, 1)
    if dehaze_method is not None and output_sharpen > 0:
        reference = base.astype(np.float32) / 255.0
        reference_blur = cv2.GaussianBlur(reference, (0, 0), 1.0)
        reference_detail = reference - reference_blur
        output_float = np.clip(
            output_float + output_sharpen * reference_detail,
            0.0,
            1.0,
        )
    output = (output_float * 255).round().astype(np.uint8)
    output = _post_fusion_color_balance(output)
    output = _luminance_stretch(output, haze)
    output = _match_exposure(output)
    output = _restore_warmth(output, img)
    output = _limit_color(output, img)
    return (output, return_inputs, weights) if return_parts else output
