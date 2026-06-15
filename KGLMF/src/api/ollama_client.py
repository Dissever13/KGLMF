"""Ollama local LLM client for second-backbone experiments."""
import torch, numpy as np, hashlib
from openai import OpenAI

class OllamaClient:
    def __init__(self, model="qwen3.5:9b", base_url="http://localhost:11434/v1", max_tokens=4096):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key="ollama", base_url=base_url)

    def generate(self, system_prompt, user_message, history=None, latent_memory=None, temperature=0.0, max_tokens=None):
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_message})
        mt = max_tokens or self.max_tokens
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=mt)
            msg = resp.choices[0].message
            content = msg.content or ""
            # Qwen3.5 is a reasoning model - check reasoning_content if content is empty
            if not content and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                content = msg.reasoning_content
            return content
        except Exception as e:
            return f"[ERROR: {e}]"

    _embedder = None
    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer
            cls._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedder

    def get_embedding(self, text):
        embedder = self._get_embedder()
        emb = embedder.encode(text[:2000], normalize_embeddings=True)
        return torch.from_numpy(emb.astype(np.float32))

    def get_logits(self, text):
        return torch.zeros(128256)
