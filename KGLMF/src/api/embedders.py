"""
Dual-embedder system for KG-LMF.
- sentence-transformers (all-MiniLM-L6-v2, d=384) for QA/reasoning tasks
- CodeBERT (microsoft/codebert-base, d=768) for code tasks

Each task type routes to the appropriate embedder for meaningful vector representations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CodeBERTEmbedder:
    """Code-specific embedder using microsoft/codebert-base.

    Uses CLS token embedding (or mean pooling) for code representation.
    Output dimension: 768 (CodeBERT hidden size).
    """

    def __init__(self, device='cpu'):
        self.device = device
        self._tokenizer = None
        self._model = None
        self.dim = 768

    def _load(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModel
            model_dir = 'D:/HF_models/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39'
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self._model = AutoModel.from_pretrained(model_dir, local_files_only=True)
            self._model.to(self.device)
            self._model.eval()

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode code text to embedding vector.

        Args:
            text: Code string (truncated to 512 tokens)
            normalize: If True, L2-normalize the output

        Returns numpy array of shape (768,)
        """
        self._load()
        # Truncate to avoid token limit
        inputs = self._tokenizer(
            text[:2000], return_tensors='pt', truncation=True,
            max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :].squeeze()

        if normalize:
            emb = F.normalize(emb, dim=0)

        return emb.cpu().numpy().astype(np.float32)

    def encode_batch(self, texts: list, normalize: bool = True) -> np.ndarray:
        """Encode a batch of code texts."""
        self._load()
        inputs = self._tokenizer(
            texts, return_tensors='pt', truncation=True,
            max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]  # CLS token

        if normalize:
            embs = F.normalize(embs, dim=1)

        return embs.cpu().numpy().astype(np.float32)


class DualEmbedder:
    """Routes embedding requests to the appropriate embedder based on task type.

    QA/reasoning → sentence-transformers (d=384)
    Code → CodeBERT (d=768)

    Both embedders normalize to unit vectors, so cosine similarity works
    across modalities within the same task type.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self._nl_embedder = None      # sentence-transformers
        self._code_embedder = None    # CodeBERT
        self.nl_dim = 384
        self.code_dim = 768

    def _get_nl(self):
        if self._nl_embedder is None:
            from sentence_transformers import SentenceTransformer
            self._nl_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._nl_embedder

    def _get_code(self):
        if self._code_embedder is None:
            self._code_embedder = CodeBERTEmbedder(device=self.device)
        return self._code_embedder

    def encode(self, text: str, task_type: str = 'qa') -> np.ndarray:
        """Encode text using the appropriate embedder for the task type."""
        if task_type == 'code':
            return self._get_code().encode(text)
        else:
            emb = self._get_nl().encode(text[:2000], normalize_embeddings=True)
            return emb.astype(np.float32)

    def encode_batch(self, texts: list, task_type: str = 'qa') -> np.ndarray:
        """Encode a batch of texts."""
        if task_type == 'code':
            return self._get_code().encode_batch(texts)
        else:
            return np.array([
                self._get_nl().encode(t[:2000], normalize_embeddings=True).astype(np.float32)
                for t in texts
            ])

    def get_dim(self, task_type: str = 'qa') -> int:
        """Get embedding dimension for task type."""
        return self.code_dim if task_type == 'code' else self.nl_dim
