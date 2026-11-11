import config
from google import genai
from google.genai import types
from PIL import Image
from typing import Union, Tuple
from ocr.provider import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        
    def generate_json(self, model_name: str, system_instruction: str, prompt_parts: list[Union[str, Image.Image, Tuple[bytes, str]]]) -> tuple[str, Union[int, None]]:
        gen_config = types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.0,
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )
        
        contents = []
        for part in prompt_parts:
            if isinstance(part, str):
                contents.append(part)
            elif isinstance(part, Image.Image):
                contents.append(part)
            elif isinstance(part, tuple) and len(part) == 2:
                data, mime_type = part
                contents.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            else:
                contents.append(part)
                
        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=gen_config
        )
        
        token_count = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_count = getattr(response.usage_metadata, "total_token_count", None)
            
        return response.text if response.text else "{}", token_count
