import torch
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEFAULT_MODEL

class DeepSeekClient:
    def __init__(self, model: str = None, api_key: str = None,
                 base_url: str = None, max_tokens: int = 4096):
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )

    def generate(self, system_prompt: str, user_message: str,
                history: List[Dict[str, str]] = None,
                latent_memory: torch.Tensor = None,
                confidence: float = None,
                temperature: float = 0.0,
                max_tokens: int = None,
                max_retries: int = 3) -> str:
        import time as _time
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-20:])
        messages.append({"role": "user", "content": user_message})

        mt = max_tokens or self.max_tokens
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=mt,
                )
                msg = response.choices[0].message
                content = msg.content or ""
                if not content and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    content = msg.reasoning_content
                return content
            except Exception as e:
                last_error = e
                err_str = str(e)
                if '502' in err_str or '503' in err_str or '429' in err_str or 'InternalServerError' in str(type(e)):
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    print(f"  [API retry {attempt+1}/{max_retries}] {type(e).__name__}, waiting {wait}s...")
                    _time.sleep(wait)
                else:
                    raise
        raise last_error

    _embedder = None
    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer
            cls._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedder

    def get_embedding(self, text: str) -> torch.Tensor:
        embedder = self._get_embedder()
        emb = embedder.encode(text[:2000], normalize_embeddings=True)
        return torch.from_numpy(emb.astype(np.float32))

    def get_logits(self, text: str) -> torch.Tensor:
        # Use chat API with logprobs to approximate logit distribution
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text[:4000]}],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            top_logprobs = resp.choices[0].logprobs.content[0].top_logprobs
            vocab_size = 128256
            logits = torch.full((vocab_size,), -1e9)
            for lp in top_logprobs:
                tid = getattr(lp, 'token_id', hash(str(getattr(lp, 'token', ''))) % vocab_size)
                if 0 <= tid < vocab_size:
                    logits[tid] = max(logits[tid].item(), getattr(lp, 'logprob', -1e9))
            return logits
        except Exception:
            # Fallback: uniform distribution for complexity estimation
            return torch.zeros(128256)
