from .composer import MemoryComposer
try:
    from .lora_utils import create_lora_config, apply_lora_to_transformer
except ImportError as _lora_import_error:
    def create_lora_config(*args, **kwargs):
        raise ImportError("LoRA utilities require peft and transformers") from _lora_import_error

    def apply_lora_to_transformer(*args, **kwargs):
        raise ImportError("LoRA utilities require peft and transformers") from _lora_import_error
