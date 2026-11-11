"""
Smart Examiner — Kraken Text Detector

Uses Kraken's BLLA segmentation model to find whole text lines.
Provides adaptive padding and filters out blank ruled lines.
"""

import logging
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List

import config
from ocr.detector import BoundingBox

logger = logging.getLogger(__name__)


class KrakenDetector:
    """Text-line detector using Kraken's segmentation engine."""

    def __init__(self):
        try:
            from kraken import blla
            self.blla = blla
            logger.info("Kraken segmentation module loaded.")
        except ImportError:
            logger.error("kraken is not installed. Please run: pip install kraken")
            raise

        # Setup debug directories
        self.debug_dir = config.DEBUG_OUTPUT_DIR
        self.kraken_crops_dir = self.debug_dir / "kraken_crops"
        self.kraken_rejected_dir = self.debug_dir / "kraken_rejected"

        self.kraken_crops_dir.mkdir(exist_ok=True, parents=True)
        self.kraken_rejected_dir.mkdir(exist_ok=True, parents=True)

    def detect(self, image: Image.Image) -> List[BoundingBox]:
        """Detect full handwritten text lines in the image.

        Args:
            image: RGB PIL Image.

        Returns:
            List of BoundingBox objects sorted in reading order.
        """
        logger.info("Running Kraken segmentation on image...")
        
        # Kraken's segment function returns a Segmentation container object
        res = self.blla.segment(image)
        
        # Kraken v7+ returns a Segmentation object with a .lines attribute
        # (list of BaselineLine objects), NOT a dictionary.
        lines = res.lines if hasattr(res, 'lines') else []
        
        if not lines:
            logger.warning("No text lines detected by Kraken.")
            return []

        logger.info(f"Kraken initially detected {len(lines)} regions.")
        
        valid_boxes = []
        rejected = 0
        accepted = 0
        
        # For debugging visual overlay
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        for i, line in enumerate(lines):
            # Each line is a BaselineLine object with a .boundary attribute
            boundary = getattr(line, 'boundary', None)
            if not boundary or len(boundary) < 3:
                continue
                
            # Convert polygon to rectangular box
            xs = [p[0] for p in boundary]
            ys = [p[1] for p in boundary]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            w = max(1.0, x_max - x_min)
            h = max(1.0, y_max - y_min)
            
            # Adaptive Padding: Add 15% vertical and 5% horizontal to prevent
            # cutting ascenders/descenders crossing ruled lines.
            pad_y = h * 0.15
            pad_x = w * 0.05
            
            box_x_min = max(0.0, x_min - pad_x)
            box_y_min = max(0.0, y_min - pad_y)
            box_x_max = min(float(image.width), x_max + pad_x)
            box_y_max = min(float(image.height), y_max + pad_y)
            
            padded_w = box_x_max - box_x_min
            padded_h = box_y_max - box_y_min
            aspect_ratio = padded_w / max(1.0, padded_h)
            
            points = [
                [box_x_min, box_y_min],
                [box_x_max, box_y_min],
                [box_x_max, box_y_max],
                [box_x_min, box_y_max]
            ]
            
            box = BoundingBox(points=points, confidence=1.0)
            
            # Filter blank ruled lines: they are extremely wide and thin
            is_ruled_line = aspect_ratio > 25
            
            # Extract crop for debugging
            crop = image.crop((box_x_min, box_y_min, box_x_max, box_y_max))
            
            if is_ruled_line:
                logger.info(f"Kraken: Line {i} rejected (AR {aspect_ratio:.2f} > 25) - Likely ruled line.")
                crop.save(self.kraken_rejected_dir / f"line_{i:03d}_ar_{aspect_ratio:.1f}.png")
                draw.rectangle([box_x_min, box_y_min, box_x_max, box_y_max], outline="red", width=2)
                rejected += 1
            else:
                valid_boxes.append(box)
                crop.save(self.kraken_crops_dir / f"line_{i:03d}.png")
                draw.rectangle([box_x_min, box_y_min, box_x_max, box_y_max], outline="green", width=3)
                accepted += 1
                
        # Save detection overlay
        overlay_path = self.debug_dir / "kraken_detections.jpg"
        draw_img.save(overlay_path)
        logger.info(f"Saved Kraken detection overlay to {overlay_path}")
        logger.info(f"Kraken accepted: {accepted}, rejected: {rejected} regions.")
        
        return valid_boxes
