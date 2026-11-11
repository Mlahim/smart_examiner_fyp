"""
Smart Examiner — Text Detector

Uses PaddleOCR in detection-only mode to find text regions.
This module is designed to be swappable — replace the implementation
without changing the interface.

Interface:
  detect(image: PIL.Image) -> list[BoundingBox]
"""

import logging
from dataclasses import dataclass
from PIL import Image
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """A detected text region on the page.

    Coordinates are in pixels relative to the input image.
    Points are ordered: top-left, top-right, bottom-right, bottom-left.
    """
    points: list[list[float]]   # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float

    @property
    def x_min(self) -> float:
        return min(p[0] for p in self.points)

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self.points)

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self.points)

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self.points)

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2


class PaddleOCRDetector:
    """Text detector using PaddleOCR's DBNet model.

    Only performs detection (no recognition). The PaddleOCR recognizer
    is disabled via `rec=False`.
    """

    def __init__(self):
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR detector...")
        self._ocr = PaddleOCR(
            lang=config.DET_MODEL_LANG,
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR detector ready.")

    def detect(self, image: Image.Image) -> list[BoundingBox]:
        """Detect text regions in the image.

        Args:
            image: RGB PIL Image.

        Returns:
            List of BoundingBox objects sorted in reading order
            (top-to-bottom, left-to-right).
        """
        img_array = np.array(image)

        # PaddleOCR returns: [page_results]
        results = self._ocr.ocr(img_array)

        if not results or not results[0]:
            logger.warning("No text detected.")
            return []

        boxes = []
        page_res = results[0]

        # Handle PaddleOCR 3.7+ dict format
        if isinstance(page_res, dict):
            polys = page_res.get('dt_polys', [])
            for poly in polys:
                pts = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
                boxes.append(BoundingBox(points=pts, confidence=1.0))
        # Handle PaddleOCR legacy list format
        else:
            for det in page_res:
                points = det[0] if isinstance(det[0], list) and isinstance(det[0][0], list) else det
                if isinstance(det, list) and len(det) == 2 and isinstance(det[1], (float, int)):
                    pts, conf = det[0], det[1]
                else:
                    pts = det if isinstance(det[0][0], (float, int)) else det[0]
                    conf = 1.0
                boxes.append(BoundingBox(points=pts, confidence=conf))

        valid_boxes = []
        for box in boxes:
            w = max(1.0, box.width)
            h = max(1.0, box.height)
            aspect_ratio = w / h
            
            # Filter extremely wide/thin boxes (like ruled lines)
            if aspect_ratio > 20:
                logger.info("Discarding extremely wide box (w:%.1f, h:%.1f, AR: %.2f)", w, h, aspect_ratio)
                continue
            
            # Filter extremely narrow/tall boxes (like vertical margins)
            if h / w > 10:
                logger.info("Discarding extremely tall box (w:%.1f, h:%.1f, AR: %.2f)", w, h, h / w)
                continue
                
            valid_boxes.append(box)

        boxes = valid_boxes

        logger.info("Detected %d valid text region(s).", len(boxes))
        return boxes
