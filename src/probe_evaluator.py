"""
Module for evaluating trained deception detection probes.
Refactored for maintainability with centralized metrics configuration.
"""

import numpy as np
import pandas as pd
import torch
import time
import os

from src import utils
from src.metrics import MetricsComputer, DEFAULT_METRICS


# Constants
DEFAULT_N_FOLDS = 5


class ProbeEvaluator:
    """Class for evaluating trained probes on test datasets."""
    
    def __init__(
        self,
        compute_control: bool = False,
        metric_names: list[str] = None,
        n_folds: int = DEFAULT_N_FOLDS,
        balance_groups: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the probe evaluator.

        Args:
            compute_control: Whether to compute control accuracies
            metric_names: List of metrics to compute (uses DEFAULT_METRICS if None)
            n_folds: Number of cross-validation folds
            balance_groups: Whether groups were balanced during training (for result tracking)
            verbose: Enable verbose output
        """
        self.compute_control = compute_control
        self.n_folds = n_folds
        self.balance_groups = balance_groups
        self.verbose = verbose
        
        # Initialize metrics computer
        self.metrics = MetricsComputer(
            metric_names=metric_names,
            compute_control=compute_control
        )
    
    @property
    def results_columns(self) -> list[str]:
        """Get column names for results dataframe."""
        return self.metrics.get_results_columns(self.n_folds)
    
    def load_or_create_results_df(self, results_csv_path: str) -> pd.DataFrame:
        """Load existing results dataframe or create a new one."""
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            required_cols = set(self.results_columns)
            existing_cols = set(existing_df.columns)
            
            if required_cols.issubset(existing_cols):
                print(f"Loaded existing results from {results_csv_path}")
                return existing_df
            
            # Check if we can add missing columns
            missing_cols = required_cols - existing_cols
            if missing_cols:
                print(f"Adding missing columns to existing results: {missing_cols}")
                for col in missing_cols:
                    existing_df[col] = np.nan
                return existing_df
        
        print("Creating new results dataframe")
        return pd.DataFrame(columns=self.results_columns)
    
    def _check_existing_result(
        self,
        results_df: pd.DataFrame,
        train_task: str,
        test_task: str,
        train_feature_type: str,
        test_feature_type: str,
        probe_type: str,
        regularization: float,
        layer_idx: int,
        use_scaler: bool,
        balance_groups: bool = False,
        is_cv_result: bool = False
    ) -> bool:
        """Check if a result already exists in the dataframe."""
        mask = (
            (results_df['train_dataset'] == train_task) &
            (results_df['test_dataset'] == test_task) &
            (results_df['train_feature_type'] == train_feature_type) &
            (results_df['test_feature_type'] == test_feature_type) &
            (results_df['probe_type'] == probe_type) &
            (results_df['regularization'] == regularization) &
            (results_df['layer_idx'] == layer_idx) &
            (results_df['use_scaler'] == use_scaler) &
            (results_df['balance_groups'] == balance_groups) &
            (results_df['is_cv_result'] == is_cv_result)
        )
        return not results_df[mask].empty
    
    def collect_probes_for_evaluation(
        self,
        train_task_list: list[str],
        test_task: str,
        layer_idx_list: list[int],
        probe_dir: str,
        train_feature_type: str,
        test_feature_type: str,
        probe_type: str,
        regularization: float,
        use_scaler: bool,
        balance_groups: bool,
        results_df: pd.DataFrame
    ) -> list[tuple]:
        """
        Collect all probes that need to be evaluated on a test task.

        Args:
            train_task_list: List of training tasks
            test_task: Test task name
            layer_idx_list: List of layer indices
            probe_dir: Directory containing probe files
            train_feature_type: Feature type used for training
            test_feature_type: Feature type used for testing
            probe_type: Type of probe
            regularization: Regularization value
            use_scaler: Whether scaler was used during training
            balance_groups: Whether groups were balanced during training
            results_df: Existing results dataframe

        Returns:
            List of tuples (train_task, layer_idx, cv_models)
        """
        probes_to_test = []

        # Determine probe filename suffixes
        scaler_suffix = "scaled" if use_scaler else "unscaled"
        balance_suffix = "balanced" if balance_groups else "unbalanced"

        for train_task in train_task_list:
            # Skip self-evaluation (handled separately for CV results)
            if test_task == train_task:
                continue

            for layer_idx in layer_idx_list:
                # Build probe path with use_scaler and balance suffixes
                task_probe_dir = f"{probe_dir}/{train_task}_{train_feature_type}"
                probe_path = f"{task_probe_dir}/probe_layer{layer_idx}_{probe_type}_{regularization}_{scaler_suffix}_{balance_suffix}.pt"
                
                if not os.path.isfile(probe_path):
                    continue

                # Load probe
                if self.verbose:
                    print(f"Loading probe for {train_task} (layer {layer_idx})")
                layer_model = torch.load(probe_path, weights_only=False)

                # Verify use_scaler matches
                metadata = layer_model.get('metadata', {})
                probe_use_scaler = metadata.get('use_scaler')

                if probe_use_scaler != use_scaler:
                    if self.verbose:
                        print(f"  Skipping: use_scaler mismatch (probe={probe_use_scaler}, requested={use_scaler})")
                    continue
                
                # Skip if result already exists
                if self._check_existing_result(
                    results_df, train_task, test_task,
                    train_feature_type, test_feature_type,
                    probe_type, regularization, layer_idx,
                    use_scaler=use_scaler,
                    balance_groups=balance_groups
                ):
                    if self.verbose:
                        print(f"Result for {train_task} -> {test_task} (layer {layer_idx}) already exists")
                    continue

                probes_to_test.append((train_task, layer_idx, layer_model))
        
        return probes_to_test
    
    def evaluate_probes_batch(
        self,
        test_task_name: str,
        layer_idx_list: list[int],
        all_probes: list[tuple],
        extracted_feats,
        all_datasets,
        tokenizer,
        test_feature_type: str
    ) -> dict:
        """
        Evaluate multiple probes on a test task efficiently.
        Processes layer by layer to minimize memory usage.
        
        Returns:
            Dictionary mapping (train_task, layer_idx) to evaluation metrics
        """
        print(f"Evaluating {len(all_probes)} probes on {test_task_name}")
        start_time = time.time()
        
        all_results = {}
        
        # Group probes by layer for efficient processing
        probes_by_layer = {}
        for train_task, layer_idx, cv_models in all_probes:
            if layer_idx not in probes_by_layer:
                probes_by_layer[layer_idx] = []
            probes_by_layer[layer_idx].append((train_task, cv_models))
        
        # Process layer by layer
        for layer_idx in layer_idx_list:
            if layer_idx not in probes_by_layer:
                continue
            
            layer_name = f'layer_{layer_idx}'
            
            if self.verbose:
                print(f"\nProcessing {layer_name}")
            
            # Load features for this layer using prepare_data
            layer_start = time.time()
            prepared_data = utils.prepare_data(
                task_name=test_task_name,
                layer_name=layer_name,
                extracted_feats=extracted_feats,
                all_datasets=all_datasets,
                tokenizer=tokenizer,
                feature_type=test_feature_type,
                balance_groups=False,
            )
            
            if not isinstance(prepared_data, utils.PreparedData):
                print(f"Warning: No valid samples for {test_task_name} at {layer_name}")
                continue

            X_test = prepared_data.X
            y_test = prepared_data.y
            
            if self.verbose:
                print(f"  Loaded features in {time.time() - layer_start:.2f}s, shape: {X_test.shape}")
            
            # Evaluate all probes for this layer
            for train_task, cv_models in probes_by_layer[layer_idx]:
                probe_results = self._evaluate_single_probe(
                    cv_models=cv_models,
                    X_test=X_test,
                    y_test=y_test
                )
                
                metadata = cv_models.get('metadata')
                use_scaler = metadata.get('use_scaler')
                
                all_results[(train_task, layer_idx, use_scaler)] = probe_results
                
                if self.verbose:
                    if not probe_results:
                        print(f"  {train_task}: No metrics computed (directions may be None in probe file)")
                    else:
                        display_metrics = {
                            name: probe_results[f'avg_{name}']
                            for name in self.metrics.metric_names
                            if f'avg_{name}' in probe_results
                        }
                        metrics_str = self.metrics.format_metrics_string(
                            display_metrics, prefix=f"  {train_task}: "
                        )
                        print(metrics_str)
        
        print(f"Evaluation completed in {time.time() - start_time:.2f}s")
        return all_results
    
    def _evaluate_single_probe(
        self,
        cv_models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """Evaluate a single probe on test data."""
        fold_metrics = {}

        metadata = cv_models.get('metadata', {})
        default_use_scaler = metadata.get('use_scaler')

        for fold_idx in range(self.n_folds):
            fold_name = f"fold_{fold_idx + 1}"
            fold_data = cv_models.get(fold_name, {})

            if not fold_data:
                continue

            # Get model parameters
            mean = fold_data.get('mean')
            scale = fold_data.get('scale')
            directions = fold_data.get('directions')
            control_threshold = fold_data.get('control_threshold', np.nan)

            # Get use_scaler from fold data, fall back to metadata
            use_scaler = fold_data.get('use_scaler', default_use_scaler)

            if directions is None:
                continue

            # Convert to numpy if needed
            if isinstance(mean, torch.Tensor):
                mean = mean.numpy()
            if isinstance(scale, torch.Tensor):
                scale = scale.numpy()
            if isinstance(directions, torch.Tensor):
                directions = directions.numpy()

            # Conditionally scale features
            if use_scaler and mean is not None and scale is not None:
                X_scaled = (X_test - mean) / scale
            else:
                # No scaling - use features directly
                X_scaled = X_test

            # Get scores - use GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_scores = torch.tensor(X_scaled, dtype=torch.float32, device=device) @ torch.tensor(directions, dtype=torch.float32, device=device)
            test_scores = test_scores.cpu().numpy()

            # Compute metrics for this fold
            metrics = self.metrics.compute_all(
                y_true=y_test,
                y_scores=test_scores,
                control_threshold=control_threshold
            )

            fold_metrics[fold_name] = metrics

        # Aggregate across folds
        if fold_metrics:
            aggregated = self.metrics.aggregate_folds(fold_metrics)
            # Include per-fold metrics in the result
            for fold_name, metrics in fold_metrics.items():
                for metric_name in self.metrics.metric_names:
                    aggregated[f'{fold_name}_{metric_name}'] = metrics.get(metric_name, np.nan)
            return aggregated

        return {}
    
    def update_results_df(
        self,
        results_df: pd.DataFrame,
        test_results: dict,
        test_task: str,
        train_feature_type: str,
        test_feature_type: str,
        model_name: str,
        probe_type: str,
        regularization: float
    ) -> pd.DataFrame:
        """Update results dataframe with new test results."""
        new_rows = []
        
        for (train_task, layer_idx, use_scaler), metrics in test_results.items():
            row = {
                'train_dataset': train_task,
                'test_dataset': test_task,
                'train_feature_type': train_feature_type,
                'test_feature_type': test_feature_type,
                'model_name': model_name,
                'layer_idx': layer_idx,
                'probe_type': probe_type,
                'regularization': regularization,
                'use_scaler': use_scaler,
                'balance_groups': self.balance_groups,
                'is_cv_result': False,
            }

            # Add metrics
            for key, value in metrics.items():
                row[key] = value

            new_rows.append(row)

            if self.verbose:
                if metrics:
                    display_metrics = {
                        name: metrics[f'avg_{name}']
                        for name in self.metrics.metric_names
                        if f'avg_{name}' in metrics
                    }
                    metrics_str = self.metrics.format_metrics_string(
                        display_metrics,
                        prefix=f"  Saving {train_task} -> {test_task} (layer {layer_idx}): "
                    )
                    print(metrics_str)
                else:
                    print(f"  Saving {train_task} -> {test_task} (layer {layer_idx}): no metrics")

        if self.verbose:
            print(f"  Added {len(new_rows)} new result rows")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            results_df = pd.concat([results_df, new_df], ignore_index=True)

        return results_df
    
    def add_cv_results(
        self,
        results_df: pd.DataFrame,
        train_task_list: list[str],
        layer_idx_list: list[int],
        probe_dir: str,
        train_feature_type: str,
        test_feature_type: str,
        probe_type: str,
        regularization: float,
        use_scaler: bool,
        balance_groups: bool,
        model_name: str
    ) -> pd.DataFrame:
        """
        Add cross-validation results from training to results dataframe.
        Each constituent dataset gets its own row for per-dataset tracking.

        This extracts CV metrics from saved probe files and adds them as
        test results where train_dataset == test_dataset.

        Args:
            results_df: Existing results dataframe
            train_task_list: List of training tasks
            layer_idx_list: List of layer indices
            probe_dir: Directory containing probe files
            train_feature_type: Feature type used for training
            test_feature_type: Feature type used for testing
            probe_type: Type of probe
            regularization: Regularization value
            use_scaler: Whether scaler was used during training
            balance_groups: Whether groups were balanced during training
            model_name: Name of the model
        """
        new_rows = []
        # Track what we've added in this batch to avoid duplicates
        added_keys = set()

        # Determine probe filename suffixes
        scaler_suffix = "scaled" if use_scaler else "unscaled"
        balance_suffix = "balanced" if balance_groups else "unbalanced"
        
        for train_task in train_task_list:
            for layer_idx in layer_idx_list:
                # Build probe path with use_scaler and balance suffixes
                task_probe_dir = f"{probe_dir}/{train_task}_{train_feature_type}"
                probe_path = f"{task_probe_dir}/probe_layer{layer_idx}_{probe_type}_{regularization}_{scaler_suffix}_{balance_suffix}.pt"
                
                if not os.path.isfile(probe_path):
                    if self.verbose:
                        print(f"Probe not found: {probe_path}")
                    continue

                # Load probe to get CV metrics
                layer_model = torch.load(probe_path, weights_only=False)
                metadata = layer_model.get('metadata', {})
                
                # Get constituent datasets (if combined task) or just the task itself
                per_dataset_metrics = metadata.get('per_dataset_metrics', {})
                constituent_datasets = metadata.get('constituent_datasets', [])
                
                if constituent_datasets and per_dataset_metrics:
                    # Combined task: save each constituent dataset as a separate row
                    for dataset_name in constituent_datasets:
                        # Create a key to track what we're adding
                        result_key = (train_task, dataset_name, layer_idx, use_scaler, balance_groups)

                        # Skip if we already added this in the current batch
                        if result_key in added_keys:
                            continue

                        # Check if result already exists in results_df
                        if self._check_existing_result(
                            results_df, train_task, dataset_name,
                            train_feature_type, test_feature_type,
                            probe_type, regularization, layer_idx,
                            use_scaler=use_scaler,
                            balance_groups=balance_groups,
                            is_cv_result=True
                        ):
                            if self.verbose:
                                print(f"CV result for {train_task} -> {dataset_name} (layer {layer_idx}) already exists")
                            continue

                        print(f"Adding CV result: {train_task} -> {dataset_name} (layer {layer_idx})")

                        row = self._create_cv_results_row_for_dataset(
                            train_task=train_task,
                            test_dataset=dataset_name,
                            train_feature_type=train_feature_type,
                            test_feature_type=test_feature_type,
                            model_name=model_name,
                            layer_idx=layer_idx,
                            probe_type=probe_type,
                            regularization=regularization,
                            use_scaler=use_scaler,
                            balance_groups=balance_groups,
                            cv_models=layer_model,
                            dataset_name=dataset_name
                        )
                        new_rows.append(row)
                        added_keys.add(result_key)

                    # Also save overall combined results, BUT only if train_task
                    # is NOT already in constituent_datasets (to avoid duplicates)
                    # FIX: This prevents duplicate rows when a single task is trained
                    # and train_task == the only constituent dataset
                    if train_task not in constituent_datasets:
                        result_key = (train_task, train_task, layer_idx, use_scaler, balance_groups)

                        if result_key not in added_keys and not self._check_existing_result(
                            results_df, train_task, train_task,
                            train_feature_type, test_feature_type,
                            probe_type, regularization, layer_idx,
                            use_scaler=use_scaler,
                            balance_groups=balance_groups,
                            is_cv_result=True
                        ):
                            print(f"Adding CV result: {train_task} -> {train_task} (overall, layer {layer_idx})")
                            row = self._create_cv_results_row_for_dataset(
                                train_task=train_task,
                                test_dataset=train_task,
                                train_feature_type=train_feature_type,
                                test_feature_type=test_feature_type,
                                model_name=model_name,
                                layer_idx=layer_idx,
                                probe_type=probe_type,
                                regularization=regularization,
                                use_scaler=use_scaler,
                                balance_groups=balance_groups,
                                cv_models=layer_model,
                                dataset_name='_overall'
                            )
                            new_rows.append(row)
                            added_keys.add(result_key)
                else:
                    # Single task: save one row
                    result_key = (train_task, train_task, layer_idx, use_scaler, balance_groups)

                    if result_key in added_keys:
                        continue

                    if self._check_existing_result(
                        results_df, train_task, train_task,
                        train_feature_type, test_feature_type,
                        probe_type, regularization, layer_idx,
                        use_scaler=use_scaler,
                        balance_groups=balance_groups,
                        is_cv_result=True
                    ):
                        continue

                    print(f"Adding CV result: {train_task} (layer {layer_idx})")
                    row = self._create_cv_results_row_for_dataset(
                        train_task=train_task,
                        test_dataset=train_task,
                        train_feature_type=train_feature_type,
                        test_feature_type=test_feature_type,
                        model_name=model_name,
                        layer_idx=layer_idx,
                        probe_type=probe_type,
                        regularization=regularization,
                        use_scaler=use_scaler,
                        balance_groups=balance_groups,
                        cv_models=layer_model,
                        dataset_name='_overall'
                    )
                    new_rows.append(row)
                    added_keys.add(result_key)
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            results_df = pd.concat([results_df, new_df], ignore_index=True)
        
        return results_df
    
    def _create_cv_results_row_for_dataset(
        self,
        train_task: str,
        test_dataset: str,
        train_feature_type: str,
        test_feature_type: str,
        model_name: str,
        layer_idx: int,
        probe_type: str,
        regularization: float,
        use_scaler: bool,
        balance_groups: bool,
        cv_models: dict,
        dataset_name: str
    ) -> dict:
        """
        Create a results row for a specific dataset's CV metrics.

        Args:
            train_task: The training task name (may be combined)
            test_dataset: The test dataset name (constituent or same as train)
            use_scaler: Whether scaler was used during training
            balance_groups: Whether groups were balanced during training
            dataset_name: Key to look up in per_dataset_metrics ('_overall' for aggregate)
            ... other standard params
        """
        metadata = cv_models.get('metadata', {})

        row = {
            'train_dataset': train_task,
            'test_dataset': test_dataset,
            'train_feature_type': train_feature_type,
            'test_feature_type': test_feature_type,
            'model_name': model_name,
            'layer_idx': layer_idx,
            'probe_type': probe_type,
            'regularization': regularization,
            'use_scaler': use_scaler,
            'balance_groups': balance_groups,
            'is_cv_result': True,  # Flag to identify CV results
        }
        
        # Get the appropriate metrics source
        per_dataset_metrics = metadata.get('per_dataset_metrics', {})
        
        if dataset_name == '_overall' or dataset_name not in per_dataset_metrics:
            # Use overall metrics from metadata
            for metric_name in self.metrics.metric_names:
                avg_key = f'avg_{metric_name}'
                std_key = f'std_{metric_name}'

                row[avg_key] = metadata.get(avg_key, np.nan)
                row[std_key] = metadata.get(std_key, np.nan)
        else:
            # Use per-dataset metrics
            ds_aggregated = per_dataset_metrics[dataset_name]
            for metric_name in self.metrics.metric_names:
                avg_key = f'avg_{metric_name}'
                std_key = f'std_{metric_name}'
                row[avg_key] = ds_aggregated.get(avg_key, np.nan)
                row[std_key] = ds_aggregated.get(std_key, np.nan)
        
        # Add per-fold metrics
        for i in range(self.n_folds):
            fold_name = f"fold_{i + 1}"
            fold_data = cv_models.get(fold_name, {})
            
            # Get metrics for this specific dataset from grouped_metrics
            grouped_metrics = fold_data.get('grouped_metrics', {})
            
            if dataset_name in grouped_metrics:
                fold_metrics = grouped_metrics[dataset_name]
            elif dataset_name == '_overall' and '_overall' in grouped_metrics:
                fold_metrics = grouped_metrics['_overall']
            else:
                # Fallback to overall fold metrics
                fold_metrics = fold_data.get('metrics', {})
            
            for metric_name in self.metrics.metric_names:
                value = fold_metrics.get(metric_name, np.nan)
                row[f'{fold_name}_{metric_name}'] = value
        
        return row