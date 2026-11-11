from ocr.gemini_provider import GeminiProvider
from ocr.provider import LLMProvider

_provider_instance = None

def get_llm_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        # In the future, this can read from config to instantiate different providers
        # e.g., if config.LLM_PROVIDER == 'ollama': return OllamaProvider()
        _provider_instance = GeminiProvider()
    return _provider_instance
