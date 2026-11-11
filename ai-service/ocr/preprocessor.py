"""
Smart Examiner — Image Preprocessor

Responsibilities:
  - Validate uploaded images (format, size, dimensions)
  - Apply conditional preprocessing (resize only when necessary)
  - Convert PDF pages to images (future)
"""

import io
import logging
from pathlib import Path
from typing import Any
from PIL import Image, ImageEnhance

import config

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when an image fails validation or preprocessing."""


def validate_image_bytes(data: bytes, filename: str = "upload") -> None:
    """Validate raw file bytes before opening.

    Checks:
      - File is not empty
      - File size is within limits
      - File extension is supported (if filename provided)
    """
    if not data:
        raise PreprocessingError("Empty file received.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise PreprocessingError(
            f"File too large: {size_mb:.1f} MB (max {config.MAX_FILE_SIZE_MB} MB)."
        )

    ext = Path(filename).suffix.lower()
    if ext and ext not in config.SUPPORTED_FORMATS:
        raise PreprocessingError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(config.SUPPORTED_FORMATS))}"
        )


def load_image(data: bytes) -> Image.Image:
    """Open raw bytes as a PIL Image and convert to RGB."""
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise PreprocessingError(f"Cannot open image: {exc}") from exc

    w, h = image.size
    if w < config.MIN_IMAGE_DIMENSION or h < config.MIN_IMAGE_DIMENSION:
        raise PreprocessingError(
            f"Image too small: {w}x{h} px "
            f"(minimum {config.MIN_IMAGE_DIMENSION}x{config.MIN_IMAGE_DIMENSION})."
        )

    return image


def preprocess(image: Image.Image) -> Image.Image:
    """Apply conditional preprocessing."""
    w, h = image.size
    max_dim = max(w, h)

    if max_dim > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Resized %dx%d → %dx%d", w, h, new_w, new_h)

    return image


def enhance_contrast_sharpness(image: Image.Image) -> Image.Image:
    """Mode: 'contrast_sharp'"""
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(1.5)
    
    enhancer_sharpness = ImageEnhance.Sharpness(image)
    image = enhancer_sharpness.enhance(2.0)
    
    logger.info("Applied contrast (1.5x) and sharpness (2.0x) enhancements.")
    return image


def remove_ruled_lines(image: Image.Image) -> Image.Image:
    """Mode: 'morphological' - Use OpenCV to detect and blur/remove horizontal ruled lines."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not installed, skipping morphological operations.")
        return image

    img_cv = np.array(image)
    if img_cv.ndim == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_cv

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines
    # Notebook lines are usually horizontal and long
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Dilate lines slightly to make sure we cover them fully
    horizontal_lines = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    # Inpaint the original image to remove the lines
    # We replace the line pixels with neighborhood pixels
    inpaint_mask = horizontal_lines
    img_cv_inpainted = cv2.inpaint(img_cv, inpaint_mask, 3, cv2.INPAINT_TELEA)

    logger.info("Applied morphological horizontal line removal.")
    return Image.fromarray(img_cv_inpainted)


def prepare_document(data: bytes, filename: str = "upload", mode: str = "none") -> tuple[Any, str]:
    """Full preprocessing pipeline for documents (PDF or Image).
    
    Returns:
        tuple containing (document_object, mime_type)
        For PDFs: returns (raw_bytes, "application/pdf")
        For Images: returns (PIL.Image, "image/jpeg")
    """
    validate_image_bytes(data, filename)
    
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        logger.info("PDF document detected, skipping image preprocessing.")
        return data, "application/pdf"
        
    image = load_image(data)
    image = preprocess(image)
    
    if mode == "contrast_sharp":
        image = enhance_contrast_sharpness(image)
    elif mode == "morphological":
        image = remove_ruled_lines(image)
    elif mode == "none":
        logger.info("No enhancement applied (mode='none').")
    else:
        logger.warning("Unknown preprocessing mode: %s. Skipping enhancement.", mode)

    logger.info("Image ready: %dx%d px", image.size[0], image.size[1])
    return image, "image/jpeg"
