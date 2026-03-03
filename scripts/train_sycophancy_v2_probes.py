#!/usr/bin/env python3
"""
Train sycophancy probes on sycophancy_v2 extracted features.

Usage:
    uv run python scripts/train_sycophancy_v2_probes.py \
        --model llama-8b \
        --features-file results/extracted_feats_all_layers_llama-8b_sycophancy_v2.h5 \
        --pairs-csv data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.probe_trainer import ProbeTrainer
from src import utils
from src.models import get_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train sycophancy v2 probes")
    parser.add_argument("--model", "-m", type=str, default="llama-8b")
    parser.add_argument("--features-file", type=str, required=True)
    parser.add_argument("--pairs-csv", type=str, required=True)
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to train on: 'all' or comma-separated (e.g. '10,15,20,25,30')")
    parser.add_argument("--probe-type", type=str, default="lr")
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--feature-type", type=str, default="last", choices=["average", "all", "last"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default="./data/huggingface/")
    args = parser.parse_args()

    # Build task name
    task_name = f"sycophancy_v2__{args.pairs_csv}"

    # Determine layers
    if args.layers == "all":
        layer_idx_list = list(range(32))
    else:
        layer_idx_list = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print("SYCOPHANCY V2 PROBE TRAINING")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Features:       {args.features_file}")
    print(f"  Pairs CSV:      {args.pairs_csv}")
    print(f"  Task name:      {task_name}")
    print(f"  Layers:         {layer_idx_list}")
    print(f"  Probe type:     {args.probe_type}")
    print(f"  Regularization: {args.regularization}")
    print(f"  Feature type:   {args.feature_type}")
    print(f"  N-folds:        {args.n_folds}")
    print("=" * 60)

    # Load features
    print(f"\nLoading features from {args.features_file}")
    extracted_feats = h5py.File(args.features_file, "r")

    # Load tokenizer only (no model needed for training)
    print("Loading tokenizer...")
    _, tokenizer = get_model_and_tokenizer(
        args.model, models_directory=args.cache_dir, omit_model=True
    )

    # Load dataset
    print("Loading dataset...")
    from src.datasets import DialogueDataset
    dataset = DialogueDataset(task_name, args.model)
    all_datasets = {task_name: dataset}
    print(f"  {len(dataset)} dialogues, group_ids present: {dataset.group_ids is not None}")

    # Train
    trainer = ProbeTrainer(
        probe_type=args.probe_type,
        regularization=args.regularization,
        n_folds=args.n_folds,
        compute_control=False,
        use_scaler=True,
        balance_groups=False,
        verbose=True,
    )

    results = trainer.train_probes_multi_layer(
        task_name=task_name,
        layer_idx_list=layer_idx_list,
        extracted_feats=extracted_feats,
        control_feats=None,
        all_datasets=all_datasets,
        tokenizer=tokenizer,
        train_feature_type=args.feature_type,
    )

    # Save
    probe_dir = f"results/probes/{args.model}/sycophancy_v2_{args.feature_type}"
    Path(probe_dir).mkdir(parents=True, exist_ok=True)
    probe_path = f"{probe_dir}/probes_{args.probe_type}_C{args.regularization}.pt"
    trainer.save_probe(results, probe_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for layer_idx in layer_idx_list:
        if layer_idx in results and "metadata" in results[layer_idx]:
            meta = results[layer_idx]["metadata"]
            auroc = meta.get("avg_auroc", float("nan"))
            auroc_std = meta.get("std_auroc", float("nan"))
            acc = meta.get("avg_accuracy", float("nan"))
            print(f"  Layer {layer_idx:2d}: AUROC={auroc:.4f} ± {auroc_std:.4f}, Acc={acc:.4f}")

    extracted_feats.close()
    print(f"\nProbes saved to {probe_path}")
    print("Done!")


if __name__ == "__main__":
    main()
