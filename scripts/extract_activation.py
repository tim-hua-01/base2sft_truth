#!/usr/bin/env python
"""
Script to extract activations from language models for various tasks.
Converted from MyExtractFeats.ipynb
"""

import os
import argparse
import h5py
import torch
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import DialogueDataset
from src.tokenized_data import TokenizedDataset
from src.models import get_model_and_tokenizer
from src.extract_act import extract_activation_nnsight

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_TASKS = [
    'claims__definitional_gemini_600_full', 'claims__evidential_gemini_600_full',
    'claims__fictional_gemini_600_full', 'claims__logical_gemini_600_full',
    'ethics__commonsense',
    'sycophancy__mmlu_stem_same_conf_all',
    'repe_honesty__IF_dishonest', 
    'roleplaying__plain',
    'insider_trading__upscale', 
    'sandbagging_v2__wmdp_mmlu',
    # 'internal_state__animals', 'internal_state__cities', 'internal_state__companies',
    # 'internal_state__elements', 'internal_state__facts', 'internal_state__inventions',
    # 'repe_honesty__plain', 
    # 'repe_honesty__IF_all', 'repe_honesty__IF_honest', 
    # 'got__best',
    # 'repe_honesty__naive', 
    # 'sycophancy__mmlu_stem_part_1', 
    # 'sycophancy__mmlu_stem_part_2',
    # 'sycophancy__mmlu_stem_conf_part_1',
    # 'sycophancy__mmlu_stem_conf_part_2',
    # 'sycophancy__mmlu_stem_conf_all',
    # 'liars_bench__harm-pressure-choice',
    # 'liars_bench__harm-pressure-knowledge-report'
]

# Model-specific layer configurations
MODEL_LAYER_CONFIGS = {
    '3b': {'num_layers': 28}, # Llama 3B
    '8b': {'num_layers': 32}, # Llama 8B
    '70b': {'num_layers': 81}, # Llama 70B
    'olmo3-7b': {'num_layers': 32}, # OLMo 3 7B (must be before '7b')
    'olmo-7b': {'num_layers': 32},  # OLMo 2 7B
    '7b': {'num_layers': 28}, # Qwen 7B (fallback)
    '14b': {'num_layers': 48}, # Qwen 14B
}


def parse_layers(layer_arg, num_layers, model_name):
    """Parse layer argument into a list of layer indices."""
    if layer_arg == 'all':
        return list(range(num_layers))
    elif layer_arg == 'sparse':
        # Every 5 layers for large models
        if 'qwen-32b' in model_name:
            return [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63]
        elif 'qwen-14b' in model_name: # for qwen 14b
            return [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]
        elif 'llama-70b' in model_name:
            return [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78]
        else:
            raise ValueError(f"do not support sparse layers for {model_name}")
    else:
        # Parse comma-separated list or range
        layers = []
        for part in layer_arg.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                layers.extend(range(start, end + 1))
            else:
                layers.append(int(part))
        return sorted(set(layers))


def extract_activations(file_name, task, layer_idx_list, dataset, model, tokenizer, batch_size=16):
        
    with h5py.File(file_name, 'a') as f:
        print(f"extracting feats for {task}, for layers {layer_idx_list}...")
        
        # create tokenized dataset
        tokenized_ds = TokenizedDataset.from_dataset(dataset, tokenizer)
        tokenized_ds.verify_detection_mask()
        
        # extract activations with detection_mask applied inside
        detection_mask = tokenized_ds.detection_mask
        extracted_feats, _ = extract_activation_nnsight(
            model, tokenized_ds, batch_size=batch_size, 
            layers=layer_idx_list, verbose=True,
            detection_mask=detection_mask  # Pass mask here
        )
        # extracted_feats shape: [total_masked_tokens, num_layers, hidden_dim]
        print(f"masked activation shape: {extracted_feats.shape}")
        
        # Verify the shape is as expected
        if len(extracted_feats.shape) != 3:
            assert False, f"Task {task} has unexpected shape {extracted_feats.shape}"
            
        layer_groups = {}
        for layer_idx in layer_idx_list:
            group_name = f'layer_{layer_idx}'
            if group_name in f:
                layer_groups[layer_idx] = f[group_name]
            else:
                layer_groups[layer_idx] = f.create_group(group_name)
                
        # Process each layer for this task
        from tqdm import tqdm
        for index, layer_idx in tqdm(enumerate(layer_idx_list)): 
            if task in layer_groups[layer_idx]:
                print(f"Dataset '{task}' already exists in layer_{layer_idx}. Overwriting...")
                del layer_groups[layer_idx][task]
                
            # Extract data for this layer: [num_samples, num_dimensions]
            layer_data = extracted_feats[:, index, :]
            
            layer_groups[layer_idx].create_dataset(task, data=layer_data, compression="gzip")


def main():
    parser = argparse.ArgumentParser(
        description='Extract activations from language models for various tasks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all layers for llama-8b (default)
  python extract_feats.py

  # Extract specific layers
  python extract_feats.py --model llama-70b-base --layers 33,37,77

  # Extract a range of layers
  python extract_feats.py --model llama-8b --layers 0-15

  # Extract all layers
  python extract_feats.py --model llama-8b --layers all

  # Extract sparse layers (every 5th layer) for large models
  python extract_feats.py --model llama-70b-base --layers sparse

  # Specify custom tasks
  python extract_feats.py --tasks sycophancy__mmlu_stem_part_1 sycophancy__mmlu_stem_part_2

  # Specify output file and cut layer
  python extract_feats.py --output ./results/custom_output.h5 --cut-at-layer 40

  python extract_feats.py --model llama-70b-base --gpus 0,1,2,3
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='llama-8b',
        help='Model name (default: llama-8b)'
    )
    
    parser.add_argument(
        '--huggingface-path', '-p',
        type=str,
        default='./data/huggingface/',
        help='Path to HuggingFace cache directory (default: ./data/huggingface/)'
    )
    
    parser.add_argument(
        '--layers', '-l',
        type=str,
        default=None,
        help='Layers to extract. Can be: "all", "sparse", comma-separated list (e.g., "33,37,77"), '
             'or range (e.g., "0-31"). Default depends on model.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output HDF5 file path. Default: ./results/extracted_feats_all_layers_{model}.h5'
    )
    
    parser.add_argument(
        '--tasks', '-t',
        type=str,
        nargs='+',
        default=None,
        help='List of tasks to process. Default: predefined task list'
    )
    
    parser.add_argument(
        '--cut-at-layer',
        type=int,
        default=None,
        help='Cut model at this layer for memory efficiency. Default: num_layers + 2'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for extraction (overrides task-specific defaults)'
    )

    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='Comma-separated list of GPU IDs to use (default: 0)'
    )
    
    args = parser.parse_args()

    # Setup GPU IDs
    if args.gpus is not None:
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else: # use default GPUs 
        gpu_ids = None
    
    # Get model configuration
    model_name = args.model
    num_layers = None
    for key in MODEL_LAYER_CONFIGS.keys():
        if key in model_name:
            num_layers = MODEL_LAYER_CONFIGS[key]['num_layers']
    if num_layers is None:
        raise ValueError(f"Model '{model_name}' not recognized for layer configuration.")
    
    # Determine layers to extract
    if args.layers is None:
        layer_idx_list = list(range(num_layers))
    else:
        layer_idx_list = parse_layers(args.layers, num_layers, model_name)
    
    # Determine output file
    if args.output is None:
        file_name = f"./results/extracted_feats_all_layers_{model_name}.h5"
    else:
        file_name = args.output
    
    # Determine tasks
    tasks = args.tasks if args.tasks is not None else DEFAULT_TASKS
    
    # Determine cut_at_layer
    cut_at_layer = args.cut_at_layer if args.cut_at_layer is not None else max(layer_idx_list) + 2
    
    huggingface_path = args.huggingface_path
    
    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: {model_name}")
    print(f"  HuggingFace path: {huggingface_path}")
    print(f"  Layers: {layer_idx_list}")
    print(f"  Output file: {file_name}")
    print(f"  Tasks: {tasks}")
    print(f"  Cut at layer: {cut_at_layer}")
    print("=" * 60)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(file_name) if os.path.dirname(file_name) else '.', exist_ok=True)
    
    # Load all datasets
    print("Loading datasets...")
    all_datasets = {}
    for dataset_name in tasks:
        all_datasets[dataset_name] = DialogueDataset(dataset_name, model_name)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(model_name, huggingface_path, 
                                               cut_at_layer=cut_at_layer, gpu_ids=gpu_ids)

    # Extract activations for each task
    bs = args.batch_size if args.batch_size is not None else 16
    for task, dataset in all_datasets.items():
        extract_activations(file_name, task, layer_idx_list, dataset, model, tokenizer, batch_size=bs)
        torch.cuda.empty_cache()

    print("Done!")


if __name__ == "__main__":
    main()