import os
import torch
from typing import Tuple, Optional, List
# from peft import PeftModel
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     PreTrainedModel,
#     PreTrainedTokenizer,
# )
from nnsight import LanguageModel

ALL_MODEL_REVISIONS: dict[str, str] = {
    'olmo3-7b-purebase': 'stage1-step1413814',
}

ALL_MODEL_PATHS = {
    # Gemma 2
    "gemma-9b": "google/gemma-2-9b-it",
    "gemma-27b": "google/gemma-2-27b-it",
    # Llama 3.0 -> for AF only
    'llama-70b-3.0': "meta-llama/Meta-Llama-3-70B-Instruct",
    'llama-70b-3.0-base': "meta-llama/Meta-Llama-3-70B",
    # Llama 3.1 
    "llama-70b-3.3": "meta-llama/Llama-3.3-70B-Instruct",
    'llama-70b-base': "meta-llama/Llama-3.1-70B",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-8b-base": "meta-llama/Llama-3.1-8B",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3b-base": "meta-llama/Llama-3.2-3B",
    

    # Qwen 2.5
    'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen-72b-base': 'Qwen/Qwen2.5-72B',
    'qwen-32b': 'Qwen/Qwen2.5-32B-Instruct',
    'qwen-32b-base': 'Qwen/Qwen2.5-32B',
    'qwen-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen-14b-base': 'Qwen/Qwen2.5-14B',
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen-7b-base': 'Qwen/Qwen2.5-7B',

    # OLMo
    'olmo-7b': 'allenai/OLMo-2-1124-7B-Instruct',
    'olmo3-7b': 'allenai/Olmo-3-7B-Instruct',
    'olmo3-7b-purebase': 'allenai/Olmo-3-1025-7B',

    # Mistral
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mistral-7b-base': 'mistralai/Mistral-7B-v0.3',
    'mistral-24b': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'mistral-24b-base': 'mistralai/Mistral-Small-24B-Base-2501',
}

def get_model_and_tokenizer(
    model_name: str,
    models_directory: str,
    omit_model: bool = False,
    gpu_ids: Optional[List[int]] = None,
    lora_path: Optional[str] = None,
    cut_at_layer: Optional[int] = None,
):
    """
    Load a Causal LM and Tokenizer with optimized settings.

    Args:
        model_name: Key from ALL_MODEL_PATHS.
        models_directory: Path to cache/store models.
        omit_model: If True, returns (None, tokenizer).
        lora_path: Path to a LoRA adapter to merge.
        cut_at_layer: Index to truncate model layers (lobotomy).
        use_flash_attention: Whether to use Flash Attention 2 (requires Ampere+ GPU).

    Returns:
        Tuple containing (Model or None, Tokenizer).
    """
    if model_name not in ALL_MODEL_PATHS:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    model_path = ALL_MODEL_PATHS[model_name]
    
    # # Load Tokenizer
    # # fast=True is generally recommended unless specific bugs arise
    # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    
    # # Ensure pad token exists. Llama/Mistral often lack a default pad token.
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.bos_token_id

    # # add simple chat template to Llama base model
    # if 'base' in model_name:
    #     tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}
    # {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}
    # {% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""
    
    # if omit_model:
    #     return None, tokenizer

    # Use float16 for all models
    dtype = torch.float16
    
    # Auto-detect device placement. 
    if gpu_ids is None:
        if "32b" in model_name or "24b" in model_name or "70b" in model_name or "72b" in model_name:
            device = "auto"
        else:
            device = "cuda:0" # use one GPU for smaller models
    else:
        if len(gpu_ids) == 1:
            device = f"cuda:{gpu_ids[0]}"
        else:
            # Multiple GPUs specified - use auto for model parallelism
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
            device = "auto"

    # Load Model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     dtype=dtype,
    #     device_map=device,
    #     cache_dir=models_directory,
    # )
    # Load Model using nnsight
    model = LanguageModel(
        model_path,
        dtype=dtype,
        device_map=device,
        cache_dir=models_directory,
    )

    # Ensure pad token exists. Llama/Mistral often lack a default pad token.
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.bos_token_id

    # Set the padding side to 'left'
    model.tokenizer.padding_side = "left"

    # Add simple chat template for base model
    if 'base' in model_name:
        model.tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}
    {% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""
    
    # # Apply LoRA (if requested)
    # if lora_path:
    #     peft_model = PeftModel.from_pretrained(model, lora_path)
    #     model = peft_model.merge_and_unload()

    # Layer Truncation
    if cut_at_layer is not None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = model.model.layers[:cut_at_layer]
        else:
            print.warning("Could not truncate: Model architecture does not match expected structure (model.model.layers).")

    if omit_model:
        return None, model.tokenizer
    else:
        return model, model.tokenizer