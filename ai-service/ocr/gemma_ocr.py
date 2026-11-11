import json
import time
import logging
from typing import Any
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from ocr.base import BaseOCR
from ocr.grouping import group_students
from ocr.teacher_grouping import group_teacher_solution

logger = logging.getLogger(__name__)

class GemmaOCR(BaseOCR):
    def __init__(self):
        # We rely on provider_factory which will check API keys
        prompt_path = config.BASE_DIR / "prompts" / "ocr_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"OCR prompt file not found at {prompt_path}")
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read().strip()
            
        logger.info(f"Gemma OCR initialized with model {config.GEMMA_MODEL_NAME}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_api(self, document: Any, mime_type: str, document_type: str = "student") -> tuple[list[dict], float, int, str]:
        """Calls the Gemini API with retry logic. Returns (parsed_pages, api_time_ms, token_count, full_json_string)."""
        start_api = time.time()
        
        base_prompt = self.prompt
        if document_type == "teacher":
            teacher_injection = (
                "\n\n*** SPECIAL INSTRUCTION FOR TEACHER RUBRICS ***\n"
                "This document is a Teacher Solution/Rubric. It contains critical grading boxes, annotations, "
                "or marginalia in the right margin (e.g., 'Explanation (2)', 'Example (2)', 'Each term (1)'). "
                "You MUST capture this text. Do NOT ignore marginalia, side-notes, or boxed text. "
                "Preserve the spatial layout roughly by placing the margin text immediately after the question or on the next line."
            )
            base_prompt += teacher_injection

        content_parts = [base_prompt]
        if mime_type == "application/pdf":
            content_parts.append((document, mime_type))
        else:
            # It's a PIL Image
            content_parts.append(document)
            
        from ocr.provider_factory import get_llm_provider
        llm = get_llm_provider()
        text, token_count = llm.generate_json(
            model_name=config.GEMMA_MODEL_NAME,
            system_instruction=(
                "You are a strict, verbatim OCR engine. You must NEVER spell-check or hallucinate words. "
                "CRITICAL RULE: If a word is crossed out, struck through, or has a line drawn through it, "
                "YOU MUST COMPLETELY OMIT IT FROM YOUR OUTPUT EVEN IF YOU CAN READ IT. "
                "Pretend it does not exist. Do not output cancelled text."
            ),
            prompt_parts=content_parts
        )
        
        api_time_ms = (time.time() - start_api) * 1000
        
        try:
            parsed_pages = json.loads(text)
            if not isinstance(parsed_pages, list):
                logger.warning("Expected JSON array from Gemini, got dict. Wrapping in list.")
                parsed_pages = [parsed_pages]
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from Gemma response. Returning empty page list.")
            parsed_pages = []
        
        # Normalize student_identity: collapse variant ID fields into canonical roll_no
        for page in parsed_pages:
            identity = page.get("student_identity")
            if isinstance(identity, dict):
                # List of variant keys that should all map to roll_no
                variant_keys = ["registration_no", "reg_no", "student_id", "id_no", "id", "enrollment_no"]
                for vk in variant_keys:
                    if vk in identity and vk != "roll_no":
                        # Only fill roll_no if it's currently empty/null
                        if not identity.get("roll_no"):
                            identity["roll_no"] = identity[vk]
                        del identity[vk]
            
        return parsed_pages, api_time_ms, token_count, text

    def extract_text(self, document: Any, save_debug: bool = True, mime_type: str = "image/jpeg", document_type: str = "student", **kwargs) -> dict[str, Any]:
        start_time = time.time()
        
        parsed_pages = []
        raw_json = ""
        api_time_ms = None
        token_count = None
        error_msg = None
        
        try:
            logger.info(f"Calling Gemini OCR API ({config.GEMMA_MODEL_NAME}) with structured JSON config...")
            
            # If it's an image, we can resize it to save bandwidth/tokens
            if mime_type != "application/pdf":
                image = document
                max_dim = 1536
                w, h = image.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image for Gemini: {w}x{h} -> {new_w}x{new_h}")
                document = image
                
            parsed_pages, api_time_ms, token_count, raw_json = self._call_api(document, mime_type, document_type)
            logger.info(f"Gemini OCR returned successfully in {api_time_ms:.0f} ms with {len(parsed_pages)} pages.")
        except Exception as e:
            logger.exception("Gemini API call failed.")
            error_msg = str(e)
            
        # Reconstruct the raw source of truth text
        full_raw_text = ""
        for page in parsed_pages:
            full_raw_text += f"--- Page {page.get('page_number', '?')} ---\n"
            full_raw_text += f"{page.get('administrative_text', '')}\n"
            full_raw_text += f"{page.get('raw_text', '')}\n\n"
            
        # Group students and questions
        grouped_data = []
        parsed_pages_for_debug = []
        if parsed_pages:
            import copy
            parsed_pages_for_debug = copy.deepcopy(parsed_pages)
            if document_type == "teacher":
                grouped_data = group_teacher_solution(parsed_pages)
            else:
                grouped_data = group_students(parsed_pages)
            
        total_time_ms = (time.time() - start_time) * 1000
        
        return {
            "engine": "gemma",
            "text": full_raw_text.strip(),
            "raw_json_string": raw_json,
            "debug_pages_pre_grouping": parsed_pages_for_debug,
            "pages": parsed_pages,
            "grouped_data": {
                "teacher_solution": grouped_data
            } if document_type == "teacher" else {
                "students": grouped_data
            },
            "processing_time_ms": total_time_ms,
            "api_response_time_ms": api_time_ms,
            "token_usage": token_count,
            "error": error_msg
        }
