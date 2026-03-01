#!/usr/bin/env python3
"""
Measure instruct model confidence on MMLU STEM questions using chat-template prompting.

Uses the same prefilled assistant turn ("I believe the best answer is (") as the
sycophancy pipeline, so confidence scores are directly comparable. Supports 0-shot
and few-shot evaluation.

Usage:
    python scripts/measure_instruct_confidence.py --model llama-8b --n-shot 0
    python scripts/measure_instruct_confidence.py --model llama-8b --n-shot 5
    python scripts/measure_instruct_confidence.py --model llama-8b --n-shot 5 --tasks abstract_algebra anatomy
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

from src.models import ALL_MODEL_PATHS, ALL_MODEL_REVISIONS

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
    return subject.replace('_', ' ')


def build_chat_prompt(
    tokenizer,
    subject: str,
    test_question: dict,
    dev_examples: list | None = None,
) -> str:
    """Build a chat-template prompt with prefilled assistant turn.

    For n-shot > 0, few-shot examples are included in the user message.
    The assistant response is prefilled with 'I believe the best answer is ('
    to match the sycophancy pipeline format.
    """
    user_content = ""

    # Few-shot examples in the user message
    if dev_examples:
        user_content += (
            f"The following are multiple choice questions (with answers) "
            f"about {format_subject(subject)}.\n\n"
        )
        for ex in dev_examples:
            user_content += f"Question: {ex['question']}\n\nChoices:\n"
            for i, choice in enumerate(ex['choices']):
                user_content += f"({LETTERS[i]}) {choice}\n"
            user_content += f"Answer: ({LETTERS[ex['answer']]})\n\n"
        user_content += "Now answer the following question.\n\n"

    # Test question
    user_content += f"Question: {test_question['question']}\n\nChoices:\n"
    for i, choice in enumerate(test_question['choices']):
        user_content += f"({LETTERS[i]}) {choice}\n"
    user_content += "\nPlease provide your answer."

    # Apply chat template to user message only, then append prefilled assistant turn
    user_only = [{'role': 'user', 'content': user_content}]
    prompt = tokenizer.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )
    prompt += "I believe the best answer is ("

    return prompt


def get_letter_token_ids(tokenizer) -> list[int]:
    """Get token IDs for A/B/C/D (computed once, reused across batches)."""
    return [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]


def get_answer_probs_batch(model, tokenizer, prompts: list[str], letter_token_ids: list[int]) -> list[dict]:
    """Extract probabilities for A/B/C/D for a batch of prompts.
    Uses left padding so position -1 is always the last real token.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", add_special_tokens=False,
        padding=True, truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, logits_to_keep=1)

    last_logits = outputs.logits[:, -1, :]

    answer_logits = last_logits[:, letter_token_ids]
    probs = torch.nn.functional.softmax(answer_logits, dim=-1)

    results = [
        {f'prob_{l}': float(probs[b, i].item()) for i, l in enumerate(LETTERS)}
        for b in range(len(prompts))
    ]

    del outputs, last_logits, inputs
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Measure instruct model confidence on MMLU STEM questions.'
    )
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Model name from registry (e.g. llama-8b)')
    parser.add_argument('--n-shot', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help='Number of few-shot examples (default: 0)')
    parser.add_argument('--tasks', '-t', type=str, nargs='+', default=None,
                        help='Subset of MMLU STEM tasks (default: all 25)')
    parser.add_argument('--cache-dir', '-c', type=str, default='./data/huggingface/',
                        help='HuggingFace cache directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for CSV (default: ./data/sycophancy_v2/<model>/instruct_confidence/)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Inference batch size (default: 64)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'./data/sycophancy_v2/{args.model}/instruct_confidence/'

    tasks = args.tasks if args.tasks else STEM_TASKS
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
    revision = ALL_MODEL_REVISIONS.get(args.model, None)
    if revision:
        print(f"Using revision: {revision}")

    if gpu_ids is None:
        device_map = "auto" if any(x in args.model for x in ("32b", "24b", "70b", "72b")) else "cuda:0"
    else:
        device_map = "auto" if len(gpu_ids) > 1 else f"cuda:{gpu_ids[0]}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir, revision=revision)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir=args.cache_dir,
        revision=revision,
    )
    model.eval()

    # Build all prompts
    print(f"Loading datasets and building {args.n_shot}-shot prompts...")
    all_prompts = []
    all_metadata = []

    for task_name in tasks:
        ds = load_dataset("cais/mmlu", task_name)
        dev_split = list(ds['dev'])
        test_split = list(ds['test'])
        fewshot_examples = dev_split[:args.n_shot] if args.n_shot > 0 else None
        print(f"  {task_name}: {len(test_split)} questions")
        for idx, question in enumerate(test_split):
            prompt = build_chat_prompt(tokenizer, task_name, question, fewshot_examples)
            all_prompts.append(prompt)
            all_metadata.append((task_name, idx, question))

    print(f"\nTotal questions: {len(all_prompts)}")

    # Run batched inference
    bs = args.batch_size
    letter_token_ids = get_letter_token_ids(tokenizer)
    all_rows = []

    for batch_start in tqdm(range(0, len(all_prompts), bs), desc="Batches"):
        batch_prompts = all_prompts[batch_start:batch_start + bs]
        batch_meta = all_metadata[batch_start:batch_start + bs]
        batch_probs = get_answer_probs_batch(model, tokenizer, batch_prompts, letter_token_ids)

        for (task_name, idx, question), probs in zip(batch_meta, batch_probs):
            correct_letter = LETTERS[question['answer']]
            model_letter = LETTERS[np.argmax([probs[f'prob_{l}'] for l in LETTERS])]
            all_rows.append({
                'id': f"{task_name}_{idx}",
                'subject': task_name,
                'model_name': args.model,
                'n_shot': args.n_shot,
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
    output_path = os.path.join(args.output_dir, f"{args.model}_{args.n_shot}shot.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Model: {args.model} | {args.n_shot}-shot")
    accuracy = (df['model_answer'] == df['correct_answer']).mean()
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Mean P(correct): {df['correct_prob'].mean():.3f}")
    high_conf = (df['correct_prob'] > 0.75).sum()
    print(f"Questions with P(correct) > 0.75: {high_conf} ({high_conf/len(df):.1%})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
