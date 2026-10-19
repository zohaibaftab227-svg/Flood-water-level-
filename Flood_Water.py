"""
water_detector.py
=====================
Flood water-level estimator (classical computer vision, OpenCV only).
No PyTorch / GPU needed. Works on photos, video files and live cameras.

IMPORTANT - HOW THIS WORKS AND ITS LIMITS
-----------------------------------------
This is a heuristic visual estimator, not a calibrated measurement.
From a single 2D image with no reference object and no camera calibration
the TRUE water depth (cm / m) cannot be known. What the tool estimates is:

    "What fraction of the vertical frame, from the bottom up, appears to be
     covered by standing water?"

That fraction is mapped to a severity category. An approximate depth in cm
is also given, but ONLY by assuming how tall the visible scene is
(`scene_height_cm`, default 150 cm). If you fix the camera in one place and
tell the tool how tall the visible scene is (e.g. a pole/wall of known
height), the cm estimate becomes much more meaningful.

Treat the output as a visual triage aid, NOT a safety-certified measurement.
"""

import time
from collections import deque

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------
def _texture_map(gray, ksize=9):
    """Local standard deviation (texture). Water tends to be smoother/flatter
    than pavement, rubble, plants, etc., so low texture is a useful clue."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f * gray_f, (ksize, ksize))
    variance = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(variance)


def _largest_bottom_component(mask):
    """Keeps only connected components of `mask` that touch the bottom rows
    of the image (flood water pools at the bottom of typical street photos),
    discarding small isolated specks elsewhere (sky reflections, etc.)."""
    h, w = mask.shape
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    bottom_band = labels[int(h * 0.9):, :]
    valid_labels = set(np.unique(bottom_band)) - {0}
    if not valid_labels:
        return np.zeros_like(mask, dtype=np.uint8)

    cleaned = np.isin(labels, list(valid_labels)).astype(np.uint8)
    return cleaned


# ----------------------------------------------------------------------
# Core detection
# ----------------------------------------------------------------------
def detect_water_mask(bgr_image, max_width=800):
    """
    Returns:
        mask       : binary uint8 mask (same size as resized image), 255 = water
        resized    : the resized BGR image the mask corresponds to
        stats      : dict with average texture / saturation inside the mask

    Uses ABSOLUTE thresholds (not Otsu) so that an image with no water at all
    is not forced into a false "water" region.
    """
    h, w = bgr_image.shape[:2]
    if w > max_width:
        scale = max_width / w
        bgr_image = cv2.resize(bgr_image, (max_width, int(h * scale)))

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    saturation = hsv[:, :, 1].astype(np.float32)
    texture = _texture_map(gray)

    texture_smoothness = np.clip(1.0 - texture / 22.0, 0, 1)   # texture>22 -> not water
    sat_score = np.clip(1.0 - saturation / 170.0, 0, 1)        # sat>170 -> not water

    combined = 0.65 * texture_smoothness + 0.35 * sat_score

    # Positional prior: flood water pools at the bottom of the frame.
    row_weight = np.linspace(0.85, 1.15, h).reshape(-1, 1)
    row_weight = np.repeat(row_weight, w, axis=1)
    combined = combined * row_weight

    binary = (combined > 0.55).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    candidate_mask = _largest_bottom_component(binary)

    stats = {"avg_texture": 0.0, "avg_saturation": 0.0, "avg_brightness": 0.0}
    if candidate_mask.sum() > 0:
        region = candidate_mask.astype(bool)
        avg_texture = float(texture[region].mean())
        if avg_texture > 16.0:
            candidate_mask = np.zeros_like(candidate_mask)
        else:
            stats["avg_texture"] = avg_texture
            stats["avg_saturation"] = float(saturation[region].mean())
            stats["avg_brightness"] = float(gray[region].mean())

    mask = candidate_mask * 255
    return mask.astype(np.uint8), bgr_image, stats


def estimate_water_level(mask):
    """
    Estimates the water level as a percentage of frame height, by finding the
    highest row where water coverage across the row is significant.

    Returns:
        level_percent : float, 0-100
        waterline_row : int, the pixel row (from top) estimated as the waterline
    """
    h, w = mask.shape
    row_coverage = (mask > 0).sum(axis=1) / w

    threshold = 0.15
    run_needed = max(3, h // 100)
    consistent_from = None

    run = 0
    for row in range(h - 1, -1, -1):
        if row_coverage[row] >= threshold:
            run += 1
            if run >= run_needed:
                consistent_from = row
        else:
            run = 0

    if consistent_from is None:
        return 0.0, h

    level_percent = ((h - consistent_from) / h) * 100
    return round(level_percent, 1), consistent_from


def categorize_level(level_percent):
    """Maps the estimated coverage percentage to a severity band.
    Returns (label, emoji, description, severity_index 0-3, bgr_color)."""
    if level_percent < 8:
        return ("Minimal / No Significant Flooding", "🟢",
                "Little to no standing water detected in the frame.",
                0, (0, 180, 0))
    elif level_percent < 25:
        return ("Low-Level Flooding (ankle-deep, approx.)", "🟡",
                "Shallow standing water - likely passable with caution on foot.",
                1, (0, 215, 255))
    elif level_percent < 55:
        return ("Moderate Flooding (knee-to-waist-deep, approx.)", "🟠",
                "Significant water coverage - vehicles and pedestrians at risk.",
                2, (0, 140, 255))
    else:
        return ("Severe Flooding (high risk)", "🔴",
                "Water covers most of the visible frame - high-risk conditions.",
                3, (0, 0, 255))


SAFETY_ADVICE = {
    0: "No action needed. Keep monitoring.",
    1: "Walk carefully; avoid driving fast through the water; watch for hidden potholes/open drains.",
    2: "Avoid crossing on foot or by small vehicles. Move valuables upward. Stay alert for rising water.",
    3: "DANGER: do not enter the water. Evacuate / move to higher ground and follow local emergency "
       "instructions (Rescue 1122 in Pakistan).",
}


# ----------------------------------------------------------------------
# Confidence + depth
# ----------------------------------------------------------------------
def _confidence(level_percent, water_area_percent, stats, mask):
    """Rough 0-100 confidence that the detected region really is water."""
    if level_percent <= 0 or water_area_percent <= 0:
        return 0.0
    smooth = np.clip(1.0 - stats["avg_texture"] / 16.0, 0, 1)         # smoother -> more water-like
    size = np.clip(water_area_percent / 12.0, 0, 1)                   # bigger region -> more reliable
    h, w = mask.shape
    bottom = (mask[int(h * 0.9):, :] > 0).mean()                      # touches the bottom strongly
    return round(float(100 * (0.45 * smooth + 0.30 * size + 0.25 * bottom)), 1)


def estimate_depth_cm(level_percent, scene_height_cm=150.0):
    """Approximate depth in cm ASSUMING the visible frame height equals
    `scene_height_cm` at the water's location. Very rough."""
    return round(level_percent / 100.0 * scene_height_cm, 1)


# ----------------------------------------------------------------------
# Full analysis of one frame
# ----------------------------------------------------------------------
def analyze_image(bgr_image, scene_height_cm=150.0, max_width=800):
    """End-to-end: image in, full result dict out (with detailed stats)."""
    mask, resized, stats = detect_water_mask(bgr_image, max_width=max_width)
    h, w = mask.shape
    level_percent, waterline_row = estimate_water_level(mask)
    label, emoji, description, severity, color = categorize_level(level_percent)

    water_pixels = int((mask > 0).sum())
    water_area_percent = round(100.0 * water_pixels / (h * w), 1)
    bottom_cov = round(100.0 * float((mask[int(h * 0.9):, :] > 0).mean()), 1)
    num_regions = max(0, cv2.connectedComponents(mask)[0] - 1)

    overlay = resized.copy()
    color_layer = np.zeros_like(resized)
    color_layer[:, :, 0] = 160   # blue tint (BGR)
    color_layer[:, :, 1] = 90
    color_layer[:, :, 2] = 0
    water_bool = mask > 0
    blended = cv2.addWeighted(resized, 0.55, color_layer, 0.45, 0)
    overlay[water_bool] = blended[water_bool]

    if level_percent > 0:
        cv2.line(overlay, (0, waterline_row), (w, waterline_row), (0, 0, 255), 2)
        # scale ticks every 25% of frame height
        for pct in (25, 50, 75):
            y = int(h * (1 - pct / 100))
            cv2.line(overlay, (0, y), (14, y), (255, 255, 255), 2)
            cv2.putText(overlay, f"{pct}%", (18, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return {
        "mask": mask,
        "resized_image": resized,
        "overlay_image": overlay,
        "level_percent": level_percent,
        "waterline_row": waterline_row,
        "frame_height_px": h,
        "frame_width_px": w,
        "water_area_percent": water_area_percent,
        "bottom_coverage_percent": bottom_cov,
        "num_water_regions": int(num_regions),
        "estimated_depth_cm": estimate_depth_cm(level_percent, scene_height_cm),
        "scene_height_cm_assumed": scene_height_cm,
        "confidence_percent": _confidence(level_percent, water_area_percent, stats, mask),
        "avg_texture_in_water": round(stats["avg_texture"], 2),
        "avg_saturation_in_water": round(stats["avg_saturation"], 1),
        "avg_brightness_in_water": round(stats["avg_brightness"], 1),
        "label": label,
        "emoji": emoji,
        "description": description,
        "severity": severity,
        "color": color,
        "advice": SAFETY_ADVICE[severity],
    }


# ----------------------------------------------------------------------
# Live-camera smoothing / trend (rising or falling water)
# ----------------------------------------------------------------------
class LevelTracker:
    """Smooths noisy per-frame levels and reports the trend
    (Rising / Falling / Stable) and rate in %-of-frame per minute."""

    def __init__(self, alpha=0.3, window_seconds=120, stable_rate=2.0):
        self.alpha = alpha
        self.window = window_seconds
        self.stable_rate = stable_rate
        self.smoothed = None
        self.history = deque()          # (timestamp, smoothed_level)
        self.peak = 0.0

    def update(self, level_percent, t=None):
        t = time.time() if t is None else t
        self.smoothed = level_percent if self.smoothed is None else (
            self.alpha * level_percent + (1 - self.alpha) * self.smoothed)
        self.history.append((t, self.smoothed))
        while self.history and t - self.history[0][0] > self.window:
            self.history.popleft()
        self.peak = max(self.peak, self.smoothed)
        return self.snapshot()

    def snapshot(self):
        rate = 0.0
        if len(self.history) >= 5:
            ts = np.array([p[0] for p in self.history])
            ls = np.array([p[1] for p in self.history])
            if ts[-1] - ts[0] >= 5:
                rate = float(np.polyfit(ts - ts[0], ls, 1)[0] * 60.0)  # % per minute
        if rate > self.stable_rate:
            trend = "Rising"
        elif rate < -self.stable_rate:
            trend = "Falling"
        else:
            trend = "Stable"
        return {
            "smoothed_level_percent": round(self.smoothed or 0.0, 1),
            "trend": trend,
            "rate_percent_per_min": round(rate, 2),
            "peak_level_percent": round(self.peak, 1),
        }


# ----------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------
def format_report(result, trend=None, title="FLOOD WATER ANALYSIS"):
    """Detailed plain-text report (one-by-one details)."""
    lines = [
        "=" * 58,
        title,
        "=" * 58,
        f"Status            : {result['emoji']} {result['label']}",
        f"Meaning           : {result['description']}",
        f"Water level       : {result['level_percent']}% of frame height (from bottom)",
        f"Approx. depth     : ~{result['estimated_depth_cm']} cm "
        f"(assumes visible scene = {result['scene_height_cm_assumed']:.0f} cm tall)",
        f"Water area        : {result['water_area_percent']}% of the whole frame",
        f"Bottom coverage   : {result['bottom_coverage_percent']}% of the bottom strip",
        f"Water regions     : {result['num_water_regions']}",
        f"Waterline row     : {result['waterline_row']} px of {result['frame_height_px']} px",
        f"Detection conf.   : {result['confidence_percent']}%",
        f"Water smoothness  : texture={result['avg_texture_in_water']} "
        f"(lower = smoother), saturation={result['avg_saturation_in_water']}, "
        f"brightness={result['avg_brightness_in_water']}",
    ]
    if trend:
        lines += [
            f"Smoothed level    : {trend['smoothed_level_percent']}%",
            f"Trend             : {trend['trend']} "
            f"({trend['rate_percent_per_min']:+.2f} %/min)",
            f"Peak level        : {trend['peak_level_percent']}%",
        ]
    lines += [
        f"Advice            : {result['advice']}",
        "-" * 58,
        "NOTE: visual estimate only - not a calibrated measurement.",
        "=" * 58,
    ]
    return "\n".join(lines)


def result_to_dict(result, trend=None):
    """JSON-safe summary (drops the image arrays)."""
    skip = {"mask", "resized_image", "overlay_image", "color"}
    d = {k: v for k, v in result.items() if k not in skip}
    if trend:
        d.update(trend)
    return d


def draw_panel(result, trend=None):
    """Returns the overlay image with an information panel drawn on top."""
    img = result["overlay_image"].copy()
    lines = [
        f"{result['label']}",
        f"Level: {result['level_percent']}%  | Depth ~{result['estimated_depth_cm']} cm",
        f"Area: {result['water_area_percent']}%  | Conf: {result['confidence_percent']}%",
    ]
    if trend:
        lines.append(f"Trend: {trend['trend']} ({trend['rate_percent_per_min']:+.1f} %/min)"
                     f" | Peak {trend['peak_level_percent']}%")
    pad, lh = 8, 20
    panel_h = pad * 2 + lh * len(lines)
    panel = img[:panel_h, :].copy()
    cv2.rectangle(panel, (0, 0), (img.shape[1], panel_h), (0, 0, 0), -1)
    img[:panel_h, :] = cv2.addWeighted(img[:panel_h, :], 0.35, panel, 0.65, 0)
    cv2.rectangle(img, (0, 0), (6, panel_h), result["color"], -1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (14, pad + lh * (i + 1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
