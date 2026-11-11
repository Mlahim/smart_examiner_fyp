from abc import ABC, abstractmethod
from typing import Any
from PIL import Image

class BaseOCR(ABC):
    """
    Abstract base class for all OCR engines.
    """
    @abstractmethod
    def extract_text(self, document: Any, **kwargs) -> dict[str, Any]:
        """
        Extracts text from the provided document (Image or PDF).
        
        Returns a dictionary with the following mandatory keys:
        - "engine": (str) The name of the OCR engine used.
        - "text": (str) The extracted plain text.
        - "processing_time_ms": (float) Total processing time.
        - "api_response_time_ms": (float | None) API call time if applicable.
        - "token_usage": (int | None) Number of tokens used if applicable.
        - "error": (str | None) Any error that occurred during extraction.
        
        Additional engine-specific metadata can also be included.
        """
        pass
