"""Direct local LLM backend with latent hidden-state injection.

This backend loads a Hugging Face/Transformers causal LM in Python and installs
forward hooks on selected decoder layers.  When ``latent_memory`` is provided to
``generate()``, each hooked layer calls ``LatentInjector.inject()`` on the live
hidden states, so KG-LMF memory affects generation without prompt text.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..injection.injector import LatentInjector


class LocalLLM:
    """Transformers-backed local LLM with latent memory layer injection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        tokenizer_path: Optional[str] = None,
        injector: Optional[LatentInjector] = None,
        d_model: int = 384,
        injection_layers: Optional[List[int]] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        max_tokens: int = 1024,
        local_files_only: bool = True,
        device_map: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LocalLLM requires transformers. Install it in the KG-LMF "
                "environment before using direct hidden-state injection."
            ) from exc

        self.model_id = model_path or model_name
        self.max_tokens = max_tokens
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.last_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.accumulated_tokens = {"prompt": 0, "completion": 0, "total": 0}
        torch_dtype = self._resolve_dtype(dtype)

        tokenizer_kwargs = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        load_kwargs = dict(tokenizer_kwargs)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        tok_id = tokenizer_path or self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_id, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_dim = int(getattr(self.model.config, "hidden_size"))
        self.injector: Optional[LatentInjector] = None
        self.set_injector(
            injector or LatentInjector(
                d_model=d_model,
                injection_layers=injection_layers or [0, 8, 16],
                hidden_dim=self.hidden_dim,
            )
        )

        self._embedder = None

    @staticmethod
    def _resolve_dtype(dtype: str):
        if dtype == "auto":
            return "auto"
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        return None

    def reset_accumulator(self) -> None:
        self.accumulated_tokens = {"prompt": 0, "completion": 0, "total": 0}

    @staticmethod
    def find_ollama_gguf_blob(
        model: str = "llama3.1:8b",
        models_dir: Optional[str] = None,
    ) -> Optional[str]:
        """Return the local Ollama GGUF blob path for a model tag, if present."""
        models_dir = models_dir or os.path.expanduser("~/.ollama/models")
        name, _, tag = model.partition(":")
        tag = tag or "latest"
        manifest = os.path.join(
            models_dir,
            "manifests",
            "registry.ollama.ai",
            "library",
            name,
            tag,
        )
        if not os.path.exists(manifest):
            return None

        with open(manifest, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for layer in data.get("layers", []):
            media_type = layer.get("mediaType", "")
            digest = layer.get("digest", "")
            if "model" not in media_type and "gguf" not in media_type:
                continue
            blob_name = digest.replace(":", "-")
            blob_path = os.path.join(models_dir, "blobs", blob_name)
            if os.path.exists(blob_path):
                return blob_path
        return None

    @staticmethod
    def _contains_hf_model_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        suffixes = (
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".gguf",
            ".safetensors.index.json",
            ".bin.index.json",
        )
        for _root, _dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith(suffixes):
                    return True
        return False

    @classmethod
    def from_hf_cache(cls, cache_dir: str, **kwargs) -> "LocalLLM":
        """Create a LocalLLM from a Hugging Face cache or direct model dir."""
        model_path = None

        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = [
                os.path.join(snapshots_dir, name)
                for name in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, name))
            ]
            snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for snapshot in snapshots:
                if cls._contains_hf_model_files(snapshot):
                    model_path = snapshot
                    break

        if model_path is None and os.path.isfile(os.path.join(cache_dir, "config.json")):
            model_path = cache_dir

        if model_path is None:
            raise FileNotFoundError(
                f"No Hugging Face model snapshot or config.json found under {cache_dir}"
            )

        kwargs["local_files_only"] = True
        return cls(model_path=model_path, **kwargs)

    @classmethod
    def from_ollama_model(cls, model: str = "llama3.1:8b", **kwargs) -> "LocalLLM":
        """Locate the Ollama blob and explain why a Transformers model is required."""
        blob_path = cls.find_ollama_gguf_blob(model)
        detail = f" Found GGUF blob: {blob_path}." if blob_path else ""
        raise RuntimeError(
            "Ollama GGUF blobs can be located, but llama.cpp does not expose "
            "decoder-layer hidden states for KG-LMF injection hooks. Use a local "
            "Hugging Face/Transformers Llama model directory with LocalLLM."
            + detail
        )

    def set_injector(self, injector: LatentInjector) -> None:
        self.injector = injector
        if hasattr(self.injector, "set_hidden_dim"):
            self.injector.set_hidden_dim(self.hidden_dim)
        self.injector.to(self._input_device())
        self.injector.eval()

    def _input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _decoder_layers(self):
        model_type = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        if "gemma" in model_type and hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if "llama" in model_type and hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        raise AttributeError("Unsupported model architecture: no decoder layer list found")

    def _install_injection_hooks(
        self,
        latent_memory: torch.Tensor,
        confidence: Optional[float],
    ):
        if self.injector is None:
            raise RuntimeError("LocalLLM requires a LatentInjector for latent memory injection")

        hooks = []
        layers = self._decoder_layers()
        for layer_idx in self.injector.injection_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                continue

            def hook(_module, _inputs, output, idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                injected = self.injector.inject(hidden, latent_memory, idx, confidence)
                if isinstance(output, tuple):
                    return (injected,) + output[1:]
                return injected

            hooks.append(layers[layer_idx].register_forward_hook(hook))
        return hooks

    def _format_prompt(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt or ""}]
        if history:
            messages.extend(history[-20:])
        messages.append({"role": "user", "content": user_message})

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        history_text = ""
        if history:
            history_text = "\n".join(
                f"{m.get('role', 'user').title()}: {m.get('content', '')}"
                for m in history[-20:]
            )
        return (
            f"System: {system_prompt or ''}\n"
            f"{history_text}\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        latent_memory: Optional[torch.Tensor] = None,
        confidence: Optional[float] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt = self._format_prompt(system_prompt, user_message, history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device())
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        hooks = []
        if latent_memory is not None:
            hooks = self._install_injection_hooks(latent_memory, confidence)

        gen_kwargs = {
            "max_new_tokens": max_tokens or self.max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            for handle in hooks:
                handle.remove()

        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        completion_tokens = int(new_tokens.shape[-1])
        self.last_tokens = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }
        for key, value in self.last_tokens.items():
            self.accumulated_tokens[key] += value
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def get_embedding(self, text: str) -> torch.Tensor:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        emb = self._embedder.encode(text[:2000], normalize_embeddings=True)
        return torch.from_numpy(emb.astype(np.float32))

    def get_logits(self, text: str) -> torch.Tensor:
        prompt = self._format_prompt("", text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device())
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, -1].detach().float().cpu()
