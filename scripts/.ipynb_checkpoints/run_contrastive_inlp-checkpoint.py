"""
Run Contrastive INLP to find type-specific to abstract directions.

This script demonstrates how to:
1. Load multi-type/task data
2. Run contrastive INLP for each type
3. Analyze which directions are most abstract vs. type-specific
4. Compare convergence across types

Usage:
    python scripts/run_contrastive_inlp.py --tasks abstract_algebra,anatomy,astronomy
"""

import argparse
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inlp_oop.contrastive_inlp import (
    run_contrastive_INLP,
    run_contrastive_INLP_all_types,
    analyze_direction_similarity,
    compute_abstraction_scores,
    find_most_abstract_directions
)


def load_task_data(task_name, data_dir='./data/activations'):
    """
    Load activation data for a task.

    Expected format:
    - X_train, y_train, X_test, y_test as numpy arrays

    Args:
        task_name: Name of the task (e.g., 'abstract_algebra')
        data_dir: Directory containing the data

    Returns:
        Dict with X_train, y_train, X_test, y_test (or X_dev, y_dev)
    """
    # Adjust this to match your actual data format
    # Example assumes .npy files or similar

    task_file = os.path.join(data_dir, f'{task_name}_activations.npy')

    if os.path.exists(task_file):
        data = np.load(task_file, allow_pickle=True).item()
        return data
    else:
        print(f"Warning: Could not find {task_file}")
        return None


def prepare_multi_task_data(task_names, data_dict=None, data_dir='./data/activations'):
    """
    Prepare data for multiple tasks in the format expected by contrastive INLP.

    Args:
        task_names: List of task names
        data_dict: Pre-loaded dictionary of {task_name: {X_train, y_train, X_test, y_test}}
                  If None, will load from data_dir
        data_dir: Directory containing data files

    Returns:
        type_data: Dict mapping task names to their data
    """
    type_data = {}

    for task in task_names:
        if data_dict is not None and task in data_dict:
            task_data = data_dict[task]
        else:
            task_data = load_task_data(task, data_dir)

        if task_data is None:
            continue

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(task_data['X_train'])
        X_test = scaler.transform(task_data['X_test'])

        # Convert to dev set naming for consistency
        type_data[task] = {
            'X_train': X_train,
            'y_train': task_data['y_train'],
            'X_dev': X_test,
            'y_dev': task_data['y_test']
        }

    return type_data


def run_analysis(
    type_data,
    num_iterations=30,
    lambda_out=1.0,
    adversarial_mode='entropy_max',
    use_neural=True,
    device='cuda',
    output_dir='./results/contrastive_inlp'
):
    """
    Run full contrastive INLP analysis.

    Args:
        type_data: Dict mapping type names to data
        num_iterations: Number of INLP iterations per type
        lambda_out: Weight for contrastive loss
        adversarial_mode: 'entropy_max', 'gradient_reversal', or 'accuracy_penalty'
        use_neural: Use neural network (True) or sklearn (False)
        device: 'cuda' or 'cpu'
        output_dir: Where to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get input dimension from first task
    first_task = list(type_data.keys())[0]
    input_dim = type_data[first_task]['X_train'].shape[1]

    print(f"\n{'='*80}")
    print(f"Running Contrastive INLP")
    print(f"{'='*80}")
    print(f"Tasks: {list(type_data.keys())}")
    print(f"Input dim: {input_dim}")
    print(f"Iterations per task: {num_iterations}")
    print(f"Lambda_out: {lambda_out}")
    print(f"Adversarial mode: {adversarial_mode}")
    print(f"Using: {'Neural' if use_neural else 'Sklearn'}")
    print(f"{'='*80}\n")

    # Run contrastive INLP for all types
    all_results = run_contrastive_INLP_all_types(
        type_data=type_data,
        num_classifiers=num_iterations,
        input_dim=input_dim,
        lambda_out=lambda_out,
        adversarial_mode=adversarial_mode,
        use_neural=use_neural,
        device=device,
        verbose=True
    )

    # Save individual results
    for task_name, results in all_results.items():
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        # Save projection matrix, directions, and metrics
        np.save(
            os.path.join(task_output_dir, f'contrastive_inlp_{task_name}.npy'),
            {
                'P': results['P'],
                'rowspace_projections': np.array(results['rowspace_projections']),
                'Ws': np.array(results['Ws']),
                'metrics': results['metrics'],
                'per_type_metrics': results['per_type_metrics']
            }
        )

        print(f"\n{task_name}: Found {len(results['Ws'])} directions")
        print(f"  Metrics saved to {task_output_dir}")

    # Analyze direction similarity across types
    print(f"\n{'='*80}")
    print("Analyzing direction similarity across types")
    print(f"{'='*80}\n")

    similarity_analysis = analyze_direction_similarity(all_results, top_k=5)

    print("Similarity matrix (last 5 directions):")
    print("  Rows/Cols:", similarity_analysis['type_names'])
    print(similarity_analysis['similarity_matrix'])

    # Find most abstract directions
    print(f"\n{'='*80}")
    print("Finding most abstract directions")
    print(f"{'='*80}\n")

    abstract_analysis = find_most_abstract_directions(all_results, type_data, n_directions=10)

    print("Top 10 most abstract directions:")
    for i, d in enumerate(abstract_analysis['most_abstract']):
        print(f"  {i+1}. Source: {d['source_type']}, "
              f"Idx: {d['direction_idx']}, "
              f"Abstraction: {d['abstraction_score']:.3f}, "
              f"Mean acc: {d['mean_accuracy']:.3f}, "
              f"Std: {d['std_accuracy']:.3f}")

    print("\nTop 10 least abstract (most type-specific) directions:")
    for i, d in enumerate(abstract_analysis['least_abstract']):
        print(f"  {i+1}. Source: {d['source_type']}, "
              f"Idx: {d['direction_idx']}, "
              f"Abstraction: {d['abstraction_score']:.3f}, "
              f"Mean acc: {d['mean_accuracy']:.3f}, "
              f"Std: {d['std_accuracy']:.3f}")

    # Save analysis results
    np.save(
        os.path.join(output_dir, 'similarity_analysis.npy'),
        similarity_analysis
    )
    np.save(
        os.path.join(output_dir, 'abstraction_analysis.npy'),
        abstract_analysis
    )

    print(f"\nAll results saved to {output_dir}")

    return all_results, similarity_analysis, abstract_analysis


def main():
    parser = argparse.ArgumentParser(description='Run Contrastive INLP')
    parser.add_argument('--tasks', type=str, required=True,
                       help='Comma-separated list of task names')
    parser.add_argument('--data-dir', type=str, default='./data/activations',
                       help='Directory containing activation data')
    parser.add_argument('--output-dir', type=str, default='./results/contrastive_inlp',
                       help='Output directory for results')
    parser.add_argument('--num-iterations', type=int, default=30,
                       help='Number of INLP iterations per task')
    parser.add_argument('--lambda-out', type=float, default=1.0,
                       help='Weight for contrastive loss (higher = more type-specific)')
    parser.add_argument('--adversarial-mode', type=str, default='entropy_max',
                       choices=['entropy_max', 'gradient_reversal', 'accuracy_penalty'],
                       help='Adversarial objective mode')
    parser.add_argument('--use-sklearn', action='store_true',
                       help='Use sklearn instead of neural network')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use')

    args = parser.parse_args()

    # Parse task names
    task_names = [t.strip() for t in args.tasks.split(',')]

    # Load data
    print(f"Loading data for tasks: {task_names}")
    type_data = prepare_multi_task_data(task_names, data_dir=args.data_dir)

    if not type_data:
        print("Error: No data loaded. Please check task names and data directory.")
        return

    print(f"Loaded {len(type_data)} tasks")
    for task, data in type_data.items():
        print(f"  {task}: train={data['X_train'].shape}, dev={data['X_dev'].shape}")

    # Run analysis
    run_analysis(
        type_data=type_data,
        num_iterations=args.num_iterations,
        lambda_out=args.lambda_out,
        adversarial_mode=args.adversarial_mode,
        use_neural=not args.use_sklearn,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
