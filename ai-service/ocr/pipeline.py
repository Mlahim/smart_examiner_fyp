"""
Smart Examiner — OCR Pipeline Orchestrator

Ties together all stages:
  Preprocess → Detect → Crop → Recognize → Reconstruct

Saves debug output at each stage for inspection.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw
import numpy as np

import config
from ocr.detector import BoundingBox
from ocr.reconstructor import TextLine, group_into_lines, merge_line_texts
from ocr.base import BaseOCR

logger = logging.getLogger(__name__)


class OCRPipeline(BaseOCR):
    """Orchestrates the full OCR pipeline.

    Components are injected via constructor, making detection
    and recognition independently swappable.
    """

    def __init__(self, detector, recognizer):
        """
        Args:
            detector: Object with a `detect(image) -> list[BoundingBox]` method.
            recognizer: Object with a `recognize(image) -> RecognitionResult` method.
        """
        self._detector = detector
        self._recognizer = recognizer
        logger.info("OCR Pipeline initialized.")

    def extract_text(self, document: Any, save_debug: bool = True, **kwargs) -> dict[str, Any]:
        """Run the full OCR pipeline on a preprocessed document.

        Args:
            document: Preprocessed RGB PIL Image.
            save_debug: Whether to save intermediate debug outputs.

        Returns:
            Standard dictionary containing extracted text and metadata.
        """
        image = document
        start_time = time.time()
        debug_dir = ""

        if save_debug:
            debug_dir = self._create_debug_dir()
            image.save(os.path.join(debug_dir, "01_original.jpg"), quality=95)

        # --- Stage 1: Detection ---
        logger.info("Stage 1: Detecting text regions...")
        boxes = self._detector.detect(image)

        if not boxes:
            elapsed = (time.time() - start_time) * 1000
            logger.warning("No text detected in image.")
            return {
                "engine": "paddle_trocr",
                "text": "",
                "processing_time_ms": elapsed,
                "api_response_time_ms": None,
                "token_usage": None,
                "error": None,
                "avg_confidence": 0.0,
                "lines_detected": 0,
                "lines_recognized": 0,
                "debug_dir": debug_dir,
            }

        # --- Stage 2: Group into lines ---
        logger.info("Stage 2: Grouping %d boxes into lines...", len(boxes))
        line_groups = group_into_lines(boxes)
        logger.info("Grouped into %d line(s).", len(line_groups))

        if save_debug:
            self._save_detection_visualization(image, boxes, line_groups, debug_dir)

        # --- Stage 3: Crop and Recognize ---
        logger.info("Stage 3: Cropping and recognizing %d line(s)...", len(line_groups))
        text_lines: list[TextLine] = []
        lines_debug_dir = ""

        if save_debug:
            lines_debug_dir = os.path.join(debug_dir, "04_lines")
            os.makedirs(lines_debug_dir, exist_ok=True)

        for idx, line_boxes in enumerate(line_groups):
            # Compute bounding rect for the entire line
            base_x_min = min(b.x_min for b in line_boxes)
            base_y_min = min(b.y_min for b in line_boxes)
            base_x_max = max(b.x_max for b in line_boxes)
            base_y_max = max(b.y_max for b in line_boxes)
            
            # Adaptive padding based on line height
            line_height = base_y_max - base_y_min
            pad = int(line_height * getattr(config, 'CROP_PADDING_RATIO', 0.15))
            
            x_min = max(0, int(base_x_min) - pad)
            y_min = max(0, int(base_y_min) - pad)
            x_max = min(image.width, int(base_x_max) + pad)
            y_max = min(image.height, int(base_y_max) + pad)

            line_crop = image.crop((x_min, y_min, x_max, y_max))

            if save_debug:
                line_crop.save(
                    os.path.join(lines_debug_dir, f"line_{idx:02d}.jpg"),
                    quality=95,
                )

            # Recognize
            result = self._recognizer.recognize(line_crop)
            avg_y = (y_min + y_max) / 2

            is_low_conf = result.confidence < config.LOW_CONF_THRESHOLD
            text_lines.append(TextLine(
                line_index=idx,
                text=result.text,
                confidence=result.confidence,
                y_position=avg_y,
                is_low_confidence=is_low_conf,
            ))

            logger.info(
                "  Line %d: conf=%.2f%%  text=%r%s",
                idx + 1, result.confidence * 100, result.text,
                " [LOW_CONF]" if is_low_conf else ""
            )

        if save_debug and lines_debug_dir:
            import json
            metadata = []
            for line in text_lines:
                metadata.append({
                    "file": f"line_{line.line_index:02d}.jpg",
                    "text": line.text,
                    "confidence": line.confidence,
                    "is_low_confidence": line.is_low_confidence
                })
            meta_path = os.path.join(lines_debug_dir, "lines_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        # --- Stage 4: Reconstruct ---
        logger.info("Stage 4: Reconstructing full text...")
        full_text = merge_line_texts(text_lines)

        if save_debug:
            result_path = os.path.join(debug_dir, "05_result.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        # Compute stats
        recognized_lines = [l for l in text_lines if l.text.strip()]
        confidences = [l.confidence for l in recognized_lines]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            "OCR complete: %d lines, avg conf %.2f%%, %.0f ms",
            len(recognized_lines), avg_conf * 100, elapsed,
        )

        return {
            "engine": "paddle_trocr",
            "text": full_text,
            "processing_time_ms": elapsed,
            "api_response_time_ms": None,
            "token_usage": None,
            "error": None,
            "avg_confidence": avg_conf,
            "lines_detected": len(line_groups),
            "lines_recognized": len(recognized_lines),
            "debug_dir": debug_dir,
        }

    # --- Debug Helpers ---

    def _create_debug_dir(self) -> str:
        """Create a timestamped debug output directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(str(config.DEBUG_OUTPUT_DIR), timestamp)
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir

    def _save_detection_visualization(
        self,
        image: Image.Image,
        boxes: list[BoundingBox],
        line_groups: list[list[BoundingBox]],
        debug_dir: str,
    ) -> None:
        """Draw raw bounding boxes and merged line boxes for debugging."""
        vis = image.copy()
        draw = ImageDraw.Draw(vis)

        # 1. Draw raw word boxes in red
        for i, box in enumerate(boxes):
            pts = [(p[0], p[1]) for p in box.points]
            draw.polygon(pts, outline="red", width=1)
            draw.text((box.x_min, max(0, box.y_min - 15)), f"w{i+1}", fill="red")

        # 2. Draw merged line boxes in blue with padding
        for i, line_boxes in enumerate(line_groups):
            base_x_min = min(b.x_min for b in line_boxes)
            base_y_min = min(b.y_min for b in line_boxes)
            base_x_max = max(b.x_max for b in line_boxes)
            base_y_max = max(b.y_max for b in line_boxes)
            
            line_height = base_y_max - base_y_min
            pad = int(line_height * getattr(config, 'CROP_PADDING_RATIO', 0.15))
            
            x_min = max(0, int(base_x_min) - pad)
            y_min = max(0, int(base_y_min) - pad)
            x_max = min(image.width, int(base_x_max) + pad)
            y_max = min(image.height, int(base_y_max) + pad)
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
            draw.text((x_min, max(0, y_min - 20)), f"Line {i+1}", fill="blue")

        vis.save(os.path.join(debug_dir, "03_detections.jpg"), quality=95)
        logger.info("Detection visualization saved. (Words: %d, Lines: %d)", len(boxes), len(line_groups))
