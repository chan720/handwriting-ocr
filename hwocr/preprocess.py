"""Image clean-up tuned for vision-language models.

VLMs read natural RGB images best, so this deliberately does NOT binarise or thin strokes the way
classic OCR preprocessing does (that throws away information the model uses). We only fix things
that genuinely hurt: rotation/skew, extreme sizes, faint ink, and heavy noise.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .config import PreprocessConfig


def _to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (1 = ink) using Otsu on a lightly blurred image."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0, step: float = 0.25) -> float:
    """Projection-profile skew estimate: the angle whose row-sum profile is 'spikiest'.

    Text lines produce sharp peaks in the horizontal projection when they are level.
    Works on a downscaled copy so it stays fast on 300-dpi scans.
    """
    h, w = gray.shape
    scale = 1000 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    mask = ink_mask(gray).astype(np.float32)
    if mask.mean() < 0.002:  # blank page
        return 0.0
    center = (mask.shape[1] / 2, mask.shape[0] / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-6, step):
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(mask, m, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1)
        score = float(np.var(profile))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 1e-3:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def resize_for_model(img: Image.Image, max_side: int, min_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        s = max_side / longest
    elif longest < min_side:
        s = min_side / longest
    else:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def enhance_contrast(img: Image.Image) -> Image.Image:
    """CLAHE on the lightness channel: darkens faint pencil without blowing out the page."""
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))


def denoise(img: Image.Image) -> Image.Image:
    arr = cv2.fastNlMeansDenoisingColored(np.asarray(img), None, 6, 6, 7, 21)
    return Image.fromarray(arr)


def preprocess(img: Image.Image, cfg: PreprocessConfig) -> tuple[Image.Image, dict]:
    """Apply the configured clean-up steps. Returns the image and a dict describing what was done."""
    info: dict = {"original_size": list(img.size)}
    img = img.convert("RGB")
    if cfg.rotate % 360:
        # PIL rotates counter-clockwise; users think in clockwise "turn the page right" terms.
        img = img.rotate(-cfg.rotate, expand=True)
        info["rotated"] = cfg.rotate
    # Work out skew on a size-normalised copy first (cheap), then rotate the full image.
    img = resize_for_model(img, max_side=max(cfg.max_side, 1), min_side=1)
    if cfg.deskew:
        angle = estimate_skew(_to_gray(img))
        if abs(angle) >= 0.3:
            img = rotate_image(img, angle)
            info["deskew_angle"] = round(angle, 2)
    if cfg.denoise:
        img = denoise(img)
        info["denoised"] = True
    if cfg.enhance_contrast:
        img = enhance_contrast(img)
        info["contrast"] = True
    img = resize_for_model(img, cfg.max_side, cfg.min_side)
    info["final_size"] = list(img.size)
    return img, info


def split_into_strips(img: Image.Image, strips: int, overlap: float) -> list[Image.Image]:
    """Split a page into horizontal strips, cutting through blank space between lines when possible.

    For each cut we search a window around the ideal position for the row with the least ink.
    If that row is genuinely empty we cut there with no overlap (no duplicated lines to merge);
    otherwise we fall back to an overlapping cut so no line is lost.
    """
    if strips <= 1:
        return [img]
    gray = _to_gray(img)
    h = gray.shape[0]
    mask = ink_mask(gray)
    # ignore a 2% side margin (page edges / binder holes produce fake "ink")
    margin = int(mask.shape[1] * 0.02)
    profile = mask[:, margin: mask.shape[1] - margin].sum(axis=1).astype(np.float32)
    profile = np.convolve(profile, np.ones(5) / 5, mode="same")
    empty_threshold = max(1.0, mask.shape[1] * 0.002)

    window = int(h / strips * 0.25)
    ov = int(h * overlap)
    bounds: list[tuple[int, int]] = []
    top = 0
    for i in range(1, strips):
        ideal = int(h * i / strips)
        lo, hi = max(top + 1, ideal - window), min(h - 1, ideal + window)
        cut = lo + int(np.argmin(profile[lo:hi])) if hi > lo else ideal
        if profile[cut] <= empty_threshold:
            bounds.append((top, cut))
            top = cut
        else:
            bounds.append((top, min(h, cut + ov // 2)))
            top = max(0, cut - ov // 2)
    bounds.append((top, h))
    return [img.crop((0, a, img.width, b)) for a, b in bounds if b - a > 10]
