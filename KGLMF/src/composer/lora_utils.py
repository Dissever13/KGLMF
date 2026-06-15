import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

def create_lora_config(rank: int = 64, alpha: int = 128,
                       target_modules: list = None) -> LoraConfig:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank, lora_alpha=alpha, lora_dropout=0.1,
        target_modules=target_modules,
    )

def apply_lora_to_transformer(model: nn.Module, config: LoraConfig) -> nn.Module:
    return get_peft_model(model, config)
