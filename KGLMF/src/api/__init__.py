try:
    from .client import DeepSeekClient
except ImportError as _openai_import_error:
    class DeepSeekClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepSeekClient requires the openai package") from _openai_import_error

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEFAULT_MODEL
from .local_llm import LocalLLM
