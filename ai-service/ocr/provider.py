from abc import ABC, abstractmethod
from typing import Any, Union, Tuple
from PIL import Image

class LLMProvider(ABC):
    @abstractmethod
    def generate_json(
        self, 
        model_name: str, 
        system_instruction: str, 
        prompt_parts: list[Union[str, Image.Image, Tuple[bytes, str]]]
    ) -> tuple[str, Union[int, None]]:
        """
        Generate JSON output from the LLM.
        prompt_parts can contain:
        - str: text prompt
        - Image.Image: PIL image
        - Tuple[bytes, str]: (file_bytes, mime_type)
        
        Returns:
            Tuple containing (response_text, token_count)
        """
        pass
