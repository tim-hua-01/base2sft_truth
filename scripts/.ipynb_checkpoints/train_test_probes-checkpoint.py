#!/usr/bin/env python3
"""
Main script for training and evaluating deception detection probes.
"""

import argparse
import os
import time
from pathlib import Path

import h5py
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probe_trainer import ProbeTrainer
from src.probe_evaluator import ProbeEvaluator
from src.metrics import DEFAULT_METRICS, METRICS_CONFIG
from config import TASK_LISTS, ALL_TASKS

from src import utils
from src.models import get_model_and_tokenizer

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate deception detection probes"
    )
    
    # Model and data arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama-70b-3.3",
        help="Model name to use (e.g., llama-70b-3.3, mistral-7b)"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        help="Path to extracted features H5 file. If not provided, will be constructed from model name and layer"
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        nargs='+',
        required=True,
        help="Layer indices to process (space-separated list)"
    )
    
    # Probe configuration
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=['lr', 'lda', 'pca', 'diffmean'],
        required=True,
        help="Type of probe to train"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        required=True,
        help="Regularization strength for logistic regression"
    )
    parser.add_argument(
        "--train-feature-type",
        type=str,
        choices=['average', 'all', 'last'],
        required=True,
        help="Feature type for training"
    )
    parser.add_argument(
        "--test-feature-type",
        type=str,
        choices=['average', 'all', 'last'],
        default='average',
        help="Feature type for testing"
    )
    
    # Scaler configuration (required)
    parser.add_argument(
        "--use-scaler",
        type=str,
        choices=['true', 'false'],
        required=True,
        help="Whether to apply StandardScaler to features before training ('true' or 'false')"
    )
    
    # Task selection
    parser.add_argument(
        "--train-tasks",
        type=str,
        nargs='+',
        help="Training tasks to use. If not specified, uses default task list"
    )
    parser.add_argument(
        "--test-tasks",
        type=str,
        nargs='+',
        help="Test tasks to use. If not specified, uses default task list"
    )
    parser.add_argument(
        "--task-preset",
        type=str,
        choices=['default', 'minimal', 'full'],
        default='default',
        help="Use preset task lists"
    )
    
    # Output configuration
    parser.add_argument(
        "--probe-dir",
        type=str,
        help="Directory to save probe models. If not provided, will be constructed from model name"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        help="Path to results CSV file. If not provided, will be constructed from model name and layer"
    )
    
    # Processing options
    parser.add_argument(
        "--compute-control",
        action='store_true',
        default=False,
        help="Compute control accuracies using alpaca dataset"
    )
    parser.add_argument(
        "--force-retrain",
        action='store_true',
        help="Force retraining even if probe files exist"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    # Metrics configuration
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        choices=list(METRICS_CONFIG.keys()),
        default=None,
        help=f"Metrics to compute. Available: {list(METRICS_CONFIG.keys())}. "
             f"Default: {DEFAULT_METRICS}"
    )
    
    # Execution options
    parser.add_argument(
        "--skip-training",
        action='store_true',
        help="Skip training phase and only run evaluation"
    )
    parser.add_argument(
        "--skip-evaluation",
        action='store_true',
        help="Skip evaluation phase and only run training"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def setup_paths(args):
    """Set up file paths based on arguments."""
    # Features file
    if args.features_file is None:
        layer_str = f"layer{args.layer_idx[0]}" if len(args.layer_idx) == 1 else "multilayer"
        args.features_file = f"./results/extracted_feats_{layer_str}_{args.model_name}.h5"
    
    # Probe directory
    if args.probe_dir is None:
        args.probe_dir = f"./results/probes/{args.model_name}"
    
    # Results CSV
    if args.results_csv is None:
        layer_str = f"layer{args.layer_idx[0]}" if len(args.layer_idx) == 1 else "multilayer"
        args.results_csv = f"./results/results_{args.model_name}_{layer_str}.csv"
    
    # Create directories
    Path(args.probe_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    
    return args


def load_task_lists(args):
    """Load task lists based on arguments."""
    preset_tasks = TASK_LISTS.get(args.task_preset, TASK_LISTS['default'])
    train_tasks = preset_tasks['train']
    test_tasks = preset_tasks['test']
    
    # Override with command line arguments if provided
    if args.train_tasks:
        train_tasks = args.train_tasks
    if args.test_tasks:
        test_tasks = args.test_tasks
    
    return train_tasks, test_tasks


def print_config(args, train_task_list, test_task_list):
    """Print configuration summary."""
    print("=" * 60)
    print("PROBE TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model:            {args.model_name}")
    print(f"  Probe type:       {args.probe_type}")
    print(f"  Regularization:   {args.regularization}")
    print(f"  Layers:           {args.layer_idx}")
    print(f"  Train features:   {args.train_feature_type}")
    print(f"  Test features:    {args.test_feature_type}")
    print(f"  Use scaler:       {args.use_scaler}")
    print(f"  N-folds:          {args.n_folds}")
    print(f"  Compute control:  {args.compute_control}")
    print(f"  Metrics:          {args.metrics or DEFAULT_METRICS}")
    print(f"  Train tasks:      {len(train_task_list)}")
    print(f"  Test tasks:       {len(test_task_list)}")
    print("=" * 60)


def train_probes(
    args,
    trainer: ProbeTrainer,
    train_task_list: list[str],
    extracted_feats,
    control_feats,
    all_datasets,
    tokenizer
):
    """Run the training phase."""
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    for train_task in train_task_list:
        print(f"\nProcessing training task: {train_task}")
        
        # Create task-specific probe directory
        task_probe_dir = f"{args.probe_dir}/{train_task}_{args.train_feature_type}"
        Path(task_probe_dir).mkdir(parents=True, exist_ok=True)
        
        # Train probes for each layer
        for layer_idx in args.layer_idx:
            # Use trainer's helper method to get consistent filename
            probe_filename = trainer.get_probe_filename(layer_idx)
            probe_path = f"{task_probe_dir}/{probe_filename}"
            
            if os.path.exists(probe_path) and not args.force_retrain:
                print(f"  Layer {layer_idx}: Probe already exists, skipping")
                continue
            
            print(f"  Training probe for layer {layer_idx}...")
            start_time = time.time()
            
            # Train probe
            cv_models = trainer.train_probe_single_layer(
                task_name=train_task,
                layer_idx=layer_idx,
                extracted_feats=extracted_feats,
                control_feats=control_feats,
                all_datasets=all_datasets,
                tokenizer=tokenizer,
                train_feature_type=args.train_feature_type,
            )
            
            if cv_models is not None:
                trainer.save_probe(cv_models, probe_path)
                print(f"  Layer {layer_idx}: Training completed in {time.time() - start_time:.2f}s")
            else:
                print(f"  Layer {layer_idx}: Training failed")


def evaluate_probes(
    args,
    evaluator: ProbeEvaluator,
    train_task_list: list[str],
    test_task_list: list[str],
    extracted_feats,
    all_datasets,
    tokenizer,
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """Run the evaluation phase."""
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    
    for test_task in test_task_list:
        print(f"\nEvaluating on test task: {test_task}")
        
        # Collect probes to test
        probes_to_test = evaluator.collect_probes_for_evaluation(
            train_task_list=train_task_list,
            test_task=test_task,
            layer_idx_list=args.layer_idx,
            probe_dir=args.probe_dir,
            train_feature_type=args.train_feature_type,
            test_feature_type=args.test_feature_type,
            probe_type=args.probe_type,
            regularization=args.regularization,
            use_scaler=args.use_scaler,
            results_df=results_df
        )
        
        if not probes_to_test:
            print("  No probes to evaluate")
            continue
        
        print(f"  Evaluating {len(probes_to_test)} probes...")
        
        # Evaluate all probes
        test_results = evaluator.evaluate_probes_batch(
            test_task_name=test_task,
            layer_idx_list=args.layer_idx,
            all_probes=probes_to_test,
            extracted_feats=extracted_feats,
            all_datasets=all_datasets,
            tokenizer=tokenizer,
            test_feature_type=args.test_feature_type
        )
        
        # Update results dataframe
        results_df = evaluator.update_results_df(
            results_df=results_df,
            test_results=test_results,
            test_task=test_task,
            train_feature_type=args.train_feature_type,
            test_feature_type=args.test_feature_type,
            model_name=args.model_name,
            probe_type=args.probe_type,
            regularization=args.regularization
        )
        
        # Save intermediate results
        results_df.to_csv(args.results_csv, index=False)
        print(f"  Results saved to {args.results_csv}")
    
    return results_df


def main():
    """Main execution function."""
    args = parse_args()
    
    # Convert use_scaler string to boolean
    args.use_scaler = args.use_scaler.lower() == 'true'
    
    args = setup_paths(args)
    
    # Load task lists
    train_task_list, test_task_list = load_task_lists(args)
    
    # Print configuration
    print_config(args, train_task_list, test_task_list)
    
    # Load features
    print(f"\nLoading features from {args.features_file}")
    extracted_feats = h5py.File(args.features_file, 'r')
    
    # Get tokenizer
    _, tokenizer = get_model_and_tokenizer(
        args.model_name,
        models_directory='../deception-detection/data/huggingface',
        omit_model=True,
    )
    
    # Get datasets
    all_datasets = utils.get_all_dataset(ALL_TASKS, args.model_name)
    # all_datasets = utils.get_all_dataset(list(set(train_task_list).union(test_task_list)), args.model_name)
    
    # Get control features if needed
    control_feats = None
    if args.compute_control:
        print("\nLoading control features from alpaca dataset...")
        control_feats = utils.get_feats_for_task(
            "alpaca__plain", f"layer_{args.layer_idx[0]}", extracted_feats
        )
    
    # Initialize trainer and evaluator with shared metric configuration
    trainer = ProbeTrainer(
        probe_type=args.probe_type,
        regularization=args.regularization,
        n_folds=args.n_folds,
        compute_control=args.compute_control,
        metric_names=args.metrics,
        use_scaler=args.use_scaler,
        verbose=args.verbose,
    )
    
    evaluator = ProbeEvaluator(
        compute_control=args.compute_control,
        metric_names=args.metrics,
        n_folds=args.n_folds,
        verbose=args.verbose
    )
    
    # Load or create results dataframe
    results_df = evaluator.load_or_create_results_df(args.results_csv)
    
    # Training phase
    if not args.skip_training:
        train_probes(
            args=args,
            trainer=trainer,
            train_task_list=train_task_list,
            extracted_feats=extracted_feats,
            control_feats=control_feats,
            all_datasets=all_datasets,
            tokenizer=tokenizer
        )
        
        # Add CV results immediately after training (before evaluation)
        # This saves per-dataset CV results as separate rows
        print("\n" + "=" * 60)
        print("SAVING CV RESULTS")
        print("=" * 60)
        
        results_df = evaluator.add_cv_results(
            results_df=results_df,
            train_task_list=train_task_list,
            layer_idx_list=args.layer_idx,
            probe_dir=args.probe_dir,
            train_feature_type=args.train_feature_type,
            test_feature_type=args.test_feature_type,
            probe_type=args.probe_type,
            regularization=args.regularization,
            use_scaler=args.use_scaler,
            model_name=args.model_name
        )
        
        # Save intermediate results
        results_df.to_csv(args.results_csv, index=False)
        print(f"CV results saved to {args.results_csv}")
    
    # Evaluation phase
    # Note: This will skip test tasks that already have results from CV
    if not args.skip_evaluation:
        results_df = evaluate_probes(
            args=args,
            evaluator=evaluator,
            train_task_list=train_task_list,
            test_task_list=test_task_list,
            extracted_feats=extracted_feats,
            all_datasets=all_datasets,
            tokenizer=tokenizer,
            results_df=results_df
        )
    
    # Final save
    results_df.to_csv(args.results_csv, index=False)
    
    # Close files
    extracted_feats.close()
    
    # Print completion summary
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {args.results_csv}")
    print(f"Probes saved to:  {args.probe_dir}")


if __name__ == "__main__":
    main()