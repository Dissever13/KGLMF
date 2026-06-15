import os

DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY", "sk-124fa3bdf68345779314dd35db4dc290")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
