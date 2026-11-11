import json
import time
import base64
import logging
from io import BytesIO
from typing import Any
from PIL import Image
import requests

import config
from ocr.base import BaseOCR

logger = logging.getLogger(__name__)


class QwenOCR(BaseOCR):
    """Qwen2.5-VL OCR engine using the same two-field JSON separation
    technique as Gemma: admin text goes into 'administrative_text',
    student answers go into 'student_answer'. Only 'student_answer'
    is returned.

    Verbatim transcription is enforced via a system message that
    acts as a hard constraint, preventing the model from auto-correcting
    spelling, grammar, or semantics.
    """

    QWEN_MAX_IMAGE_DIM = 1024  # Resize images before sending to the VLM

    def __init__(self):
        self.api_url = config.OLLAMA_API_URL.replace("/api/generate", "/api/chat")
        self.model_name = config.QWEN_MODEL_NAME

        # System prompt: enforces verbatim transcription as an immutable constraint
        system_prompt_path = config.BASE_DIR / "prompts" / "qwen_system_prompt.txt"
        if not system_prompt_path.exists():
            raise FileNotFoundError(f"Qwen system prompt not found at {system_prompt_path}")
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

        # User prompt: instructs the model to separate admin vs student text into JSON
        user_prompt_path = config.BASE_DIR / "prompts" / "qwen_ocr_prompt.txt"
        if not user_prompt_path.exists():
            raise FileNotFoundError(f"Qwen OCR prompt not found at {user_prompt_path}")
        with open(user_prompt_path, "r", encoding="utf-8") as f:
            self.user_prompt = f.read().strip()

        logger.info(f"Qwen OCR initialized with model {self.model_name} pointing to {self.api_url}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Resize (if needed) and convert a PIL Image to a base64 string for the Ollama API."""
        w, h = image.size
        max_dim = max(w, h)
        if max_dim > self.QWEN_MAX_IMAGE_DIM:
            scale = self.QWEN_MAX_IMAGE_DIM / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Resized image for Qwen: {w}x{h} → {new_w}x{new_h}")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_api(self, image: Image.Image) -> tuple[str, float, int, str]:
        """Calls the local Ollama chat API with system + user messages.
        Returns (student_text, api_time_ms, eval_count, raw_response).
        """
        start_api = time.time()

        b64_image = self._image_to_base64(image)

        # Two-message approach:
        # 1. System message: hard constraint for verbatim transcription
        # 2. User message: task instructions with the image
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.user_prompt,
                    "images": [b64_image],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
            }
        }

        # Generous timeout: 3B VLM on CPU can take 2-5 minutes per image.
        response = requests.post(self.api_url, json=payload, timeout=300)

        # Log error body before raising so we can debug Ollama errors
        if response.status_code != 200:
            logger.error(f"Ollama returned {response.status_code}: {response.text[:500]}")
            response.raise_for_status()

        data = response.json()

        api_time_ms = (time.time() - start_api) * 1000
        raw_text = data.get("message", {}).get("content", "")

        # The model is now operating in pure raw text mode.
        # No JSON parsing is required.
        combined_answer = raw_text.strip()
        
        return combined_answer, api_time_ms, data.get("eval_count"), raw_text

    def extract_text(self, document: Any, save_debug: bool = True, **kwargs) -> dict[str, Any]:
        """Run OCR using local Qwen2.5-VL via Ollama.
        
        Args:
            document: Preprocessed RGB PIL Image.
            save_debug: Whether to save intermediate debug outputs.
            kwargs: Any additional arguments.
        """
        image = document
        start_time = time.time()
        text = ""
        raw_response = ""
        api_time_ms = None
        token_count = None
        error_msg = None

        try:
            logger.info(f"Calling Qwen OCR API ({self.model_name}) — verbatim transcription mode...")
            text, api_time_ms, token_count, raw_response = self._call_api(image)
            logger.info(f"Qwen OCR returned successfully in {api_time_ms:.0f} ms.")
        except Exception as e:
            logger.exception("Qwen API call failed.")
            error_msg = str(e)

        total_time_ms = (time.time() - start_time) * 1000

        return {
            "engine": "qwen",
            "text": text,
            "raw_json": raw_response,
            "processing_time_ms": total_time_ms,
            "api_response_time_ms": api_time_ms,
            "token_usage": token_count,
            "error": error_msg,
            "cropped_image": image,
        }
