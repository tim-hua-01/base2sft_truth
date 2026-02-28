#!/usr/bin/env python3
"""
Measure base model confidence on MMLU STEM questions using 5-shot prompting.

Uses standard MMLU 5-shot text-completion format (no chat template) to extract
P(A), P(B), P(C), P(D) from base models. Saves per-question probabilities
for downstream filtering (e.g. keep questions where P(correct) > 0.75).

Usage:
    python scripts/measure_base_confidence.py --model llama-8b-base
    python scripts/measure_base_confidence.py --model llama-8b-base --tasks abstract_algebra anatomy
    python scripts/measure_base_confidence.py --model llama-8b-base --cache-dir ./data/huggingface/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.models import ALL_MODEL_PATHS

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

STEM_TASKS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_medicine', 'college_physics',
    'computer_security', 'conceptual_physics', 'econometrics',
    'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_mathematics',
    'high_school_physics', 'high_school_statistics',
    'machine_learning', 'medical_genetics', 'virology'
]

LETTERS = ['A', 'B', 'C', 'D']


def format_subject(subject: str) -> str:
    """Convert task name to readable subject (e.g. 'abstract_algebra' -> 'abstract algebra')."""
    return subject.replace('_', ' ')


def build_fewshot_prompt(subject: str, dev_examples: list, test_question: dict) -> str:
    """Build the standard 5-shot MMLU text-completion prompt."""
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"

    # Add dev examples (up to 5)
    for ex in dev_examples:
        prompt += f"{ex['question']}\n"
        for i, choice in enumerate(ex['choices']):
            prompt += f"{LETTERS[i]}. {choice}\n"
        prompt += f"Answer: {LETTERS[ex['answer']]}\n\n"

    # Add test question (no answer)
    prompt += f"{test_question['question']}\n"
    for i, choice in enumerate(test_question['choices']):
        prompt += f"{LETTERS[i]}. {choice}\n"
    prompt += "Answer:"

    return prompt


def get_answer_probs_batch(model, tokenizer, prompts: list[str]) -> list[dict]:
    """Extract probabilities for A/B/C/D for a batch of prompts.
    Uses left padding so position -1 is always the last real token for each sequence.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", add_special_tokens=True,
        padding=True, truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, logits_to_keep=1)

    last_logits = outputs.logits[:, -1, :]  # (batch, vocab) - only last token computed

    token_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    answer_logits = last_logits[:, token_ids]  # (batch, 4)
    probs = torch.nn.functional.softmax(answer_logits, dim=-1)

    results = [{f'prob_{l}': float(probs[b, i].item()) for i, l in enumerate(LETTERS)}
               for b in range(len(prompts))]

    del outputs, last_logits, inputs
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Measure base model confidence on MMLU STEM questions.'
    )
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Model name from registry (e.g. llama-8b-base)')
    parser.add_argument('--tasks', '-t', type=str, nargs='+', default=None,
                        help='Subset of MMLU STEM tasks (default: all 25)')
    parser.add_argument('--cache-dir', '-c', type=str, default='./data/huggingface/',
                        help='HuggingFace cache directory')
    parser.add_argument('--output-dir', type=str, default='./data/sycophancy_v2/base_confidence/',
                        help='Output directory for CSV')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Inference batch size (default: 64)')
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else STEM_TASKS
    # Validate task names
    for t in tasks:
        if t not in STEM_TASKS:
            print(f"Warning: '{t}' not in STEM_TASKS list, proceeding anyway")

    # GPU setup
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Load model
    print(f"Loading model: {args.model}")
    if args.model not in ALL_MODEL_PATHS:
        raise ValueError(f"Model '{args.model}' not found in registry.")
    model_path = ALL_MODEL_PATHS[args.model]

    if gpu_ids is None:
        device_map = "auto" if any(x in args.model for x in ("32b", "24b", "70b", "72b")) else "cuda:0"
    else:
        device_map = "auto" if len(gpu_ids) > 1 else f"cuda:{gpu_ids[0]}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir=args.cache_dir,
    )
    model.eval()

    # Build all prompts and metadata upfront across all tasks
    print("Loading datasets and building prompts...")
    all_prompts = []
    all_metadata = []  # (task_name, per_task_idx, question)

    for task_name in tasks:
        ds = load_dataset("cais/mmlu", task_name)
        dev_split = list(ds['dev'])
        test_split = list(ds['test'])
        fewshot_examples = dev_split[:5]
        print(f"  {task_name}: {len(test_split)} questions")
        for idx, question in enumerate(test_split):
            all_prompts.append(build_fewshot_prompt(task_name, fewshot_examples, question))
            all_metadata.append((task_name, idx, question))

    print(f"\nTotal questions: {len(all_prompts)}")

    # Run inference in batches across all tasks
    bs = args.batch_size
    all_rows = []

    for batch_start in tqdm(range(0, len(all_prompts), bs), desc="Batches"):
        batch_prompts = all_prompts[batch_start:batch_start + bs]
        batch_meta = all_metadata[batch_start:batch_start + bs]
        batch_probs = get_answer_probs_batch(model, tokenizer, batch_prompts)

        for (task_name, idx, question), probs in zip(batch_meta, batch_probs):
            correct_letter = LETTERS[question['answer']]
            model_letter = LETTERS[np.argmax([probs[f'prob_{l}'] for l in LETTERS])]
            all_rows.append({
                'id': f"{task_name}_{idx}",
                'subject': task_name,
                'model_name': args.model,
                'correct_answer': correct_letter,
                'model_answer': model_letter,
                'correct_prob': probs[f'prob_{correct_letter}'],
                'prob_A': probs['prob_A'],
                'prob_B': probs['prob_B'],
                'prob_C': probs['prob_C'],
                'prob_D': probs['prob_D'],
            })

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model}.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Saved {len(df)} rows to {output_path}")
    accuracy = (df['model_answer'] == df['correct_answer']).mean()
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Mean P(correct): {df['correct_prob'].mean():.3f}")
    high_conf = (df['correct_prob'] > 0.75).sum()
    print(f"Questions with P(correct) > 0.75: {high_conf} ({high_conf/len(df):.1%})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
