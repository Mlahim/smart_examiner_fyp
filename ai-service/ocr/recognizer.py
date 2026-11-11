"""
Smart Examiner — Text Recognizer

Uses Microsoft TrOCR (Large Handwritten) for line-level text recognition.
This module is designed to be swappable — replace the implementation
without changing the interface.

Interface:
  recognize(line_image: PIL.Image) -> RecognitionResult
"""

import logging
from dataclasses import dataclass
from PIL import Image
import torch

import config

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Result from recognizing a single text line."""
    text: str
    confidence: float           # 0.0 to 1.0


class TrOCRRecognizer:
    """Text recognizer using Microsoft TrOCR (Large Handwritten).

    Expects cropped single-line images as input.
    """

    def __init__(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._device = torch.device(config.TROCR_DEVICE)
        logger.info(
            "Loading TrOCR model from '%s' on %s...",
            config.TROCR_MODEL_NAME, self._device,
        )

        self._processor = TrOCRProcessor.from_pretrained(
            config.TROCR_MODEL_NAME, use_fast=False,
        )
        self._model = VisionEncoderDecoderModel.from_pretrained(
            config.TROCR_MODEL_NAME,
        ).to(self._device)
        self._model.eval()

        logger.info("TrOCR model loaded successfully.")

    def recognize(self, line_image: Image.Image) -> RecognitionResult:
        """Recognize text from a single-line crop.

        Args:
            line_image: RGB PIL Image containing one line of text.

        Returns:
            RecognitionResult with extracted text and confidence.
        """
        pixel_values = self._processor(
            images=line_image, return_tensors="pt",
        ).pixel_values.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=config.TROCR_MAX_NEW_TOKENS,
            )

        # Decode text
        text = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=True,
        )[0].strip()

        # Confidence: average of per-token max softmax probabilities
        scores = outputs.scores
        if scores:
            confs = [torch.softmax(s, dim=-1).max().item() for s in scores]
            avg_conf = sum(confs) / len(confs)
        else:
            avg_conf = 0.0

        return RecognitionResult(text=text, confidence=avg_conf)
