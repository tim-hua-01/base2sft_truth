"""
Module for training deception detection probes.
Refactored for maintainability with centralized metrics configuration.
"""

import numpy as np
import torch
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

from src import utils, probes
from src.utils import PreparedData
from src.metrics import MetricsComputer, DEFAULT_METRICS


# Constants
DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_STATE = 42


class ProbeTrainer:
    """Class for training various types of probes."""
    
    def __init__(
        self,
        probe_type: str = 'lr',
        regularization: float = 1.0,
        n_folds: int = DEFAULT_N_FOLDS,
        compute_control: bool = True,
        metric_names: list[str] = None,
        use_scaler: bool = True,
        balance_groups: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the probe trainer.

        Args:
            probe_type: Type of probe ('lr', 'lda', 'pca', 'diffmean')
            regularization: Regularization strength for LR
            n_folds: Number of cross-validation folds
            compute_control: Whether to compute control accuracies
            metric_names: List of metrics to compute (uses DEFAULT_METRICS if None)
            use_scaler: Whether to apply StandardScaler to features
            balance_groups: Whether to balance group sizes in combined datasets
            verbose: Enable verbose output
        """
        self.probe_type = probe_type
        self.regularization = regularization
        self.n_folds = n_folds
        self.compute_control = compute_control
        self.use_scaler = use_scaler
        self.balance_groups = balance_groups
        self.verbose = verbose
        
        # Initialize metrics computer
        self.metrics = MetricsComputer(
            metric_names=metric_names,
            compute_control=compute_control
        )
    
    def get_probe_model(self):
        """Get the appropriate probe model based on probe_type."""
        probe_factories = {
            'lr': lambda: probes.LogisticRegressionProbe(self.regularization),
            'diffmean': lambda: probes.DifferenceOfMeansProbe(),
            'pca': lambda: probes.PCAProbe(),
            'lda': lambda: probes.LDAProbe(),
        }
        
        if self.probe_type not in probe_factories:
            raise ValueError(f"Unknown probe type: {self.probe_type}. "
                           f"Available: {list(probe_factories.keys())}")
        
        return probe_factories[self.probe_type]()
    
    def train_probe_single_layer(
        self,
        task_name: str,
        layer_idx: int,
        extracted_feats,
        control_feats,
        all_datasets,
        tokenizer,
        train_feature_type: str
    ) -> dict | None:
        """
        Train a probe for a specific layer using cross-validation.
        
        Returns:
            Dictionary containing models and metrics for each fold, or None if no data
        """
        if self.verbose:
            print(f"Training {self.probe_type} probe for {task_name}, layer={layer_idx}")
        
        # Prepare data
        layer_name = f"layer_{layer_idx}"
        prepared = utils.prepare_data(
            task_name, layer_name, extracted_feats, all_datasets,
            tokenizer, balance_groups=self.balance_groups, feature_type=train_feature_type
        )
        
        if not isinstance(prepared, utils.PreparedData):
            print(f"No valid samples found for {task_name}, layer={layer_idx}")
            return None

        data = prepared
        
        if self.verbose:
            print(f"Data shape: X={data.X.shape}, y={data.y.shape}")
            print(f"Datasets: {list(data.dataset_names.values())}")
        
        # Prepare cross-validation splits
        fold_splits = self._get_cv_splits(data.X, data.y)
        
        # Get control features for this layer
        layer_control_feats = (
            control_feats[:, layer_idx, :] 
            if control_feats is not None else None
        )
        
        # Train on each fold
        cv_models = {}
        fold_metrics_list = []
        fold_grouped_metrics = {}  # For per-dataset tracking
        
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            fold_name = f"fold_{fold_idx + 1}"
            
            fold_results = self._train_fold(
                X_train=data.X[train_idx],
                X_test=data.X[test_idx],
                y_train=data.y[train_idx],
                y_test=data.y[test_idx],
                dataset_ids_test=data.dataset_ids[test_idx],
                dataset_names=data.dataset_names,
                control_feats=layer_control_feats
            )
            
            if fold_results is not None:
                cv_models[fold_name] = fold_results
                fold_metrics_list.append(fold_results['metrics'])
                fold_grouped_metrics[fold_name] = fold_results['grouped_metrics']
                
                if self.verbose:
                    metrics_str = self.metrics.format_metrics_string(
                        fold_results['metrics'], prefix=f"  {fold_name}: "
                    )
                    print(f"{metrics_str}, Train Acc={fold_results['train_accuracy']:.4f}")
                    
                    # Print per-dataset metrics
                    for ds_name, ds_metrics in fold_results['grouped_metrics'].items():
                        if ds_name != '_overall':
                            ds_str = self.metrics.format_metrics_string(ds_metrics, prefix=f"    {ds_name}: ")
                            print(ds_str)
        
        # Aggregate metrics across folds
        fold_metrics_dict = {
            f"fold_{i+1}": m for i, m in enumerate(fold_metrics_list)
        }
        aggregated = self.metrics.aggregate_folds(fold_metrics_dict)
        
        # Aggregate per-dataset metrics across folds
        aggregated_per_dataset = self.metrics.aggregate_grouped_folds(fold_grouped_metrics)
        
        # Print summary
        self._print_cv_summary(aggregated, aggregated_per_dataset)
        
        # Store metadata
        cv_models['metadata'] = self._create_metadata(
            task_name, layer_idx, train_feature_type, aggregated,
            aggregated_per_dataset, data.dataset_names
        )
        
        return cv_models
    
    def _get_cv_splits(self, X, y):
        """Get cross-validation splits."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        return list(kf.split(X))
    
    def _train_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        control_feats,
        dataset_ids_test: np.ndarray = None,
        dataset_names: dict[int, str] = None
    ) -> dict | None:
        """Train and evaluate a single fold with per-dataset metrics."""
        # Conditionally apply StandardScaler
        if self.use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            scaler_mean = torch.tensor(scaler.mean_)
            scaler_scale = torch.tensor(scaler.scale_)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            scaler_mean = torch.zeros(X_train.shape[1])
            scaler_scale = torch.ones(X_train.shape[1])
        
        # Train model
        model = self.get_probe_model()
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Get predictions and scores
        train_predictions = model.predict(X_train_scaled)
        train_accuracy = np.mean(train_predictions == y_train)
        test_scores = self._get_decision_scores(model, X_test_scaled)
        
        # Calculate control threshold if applicable
        control_threshold = self._compute_control_threshold(
            scaler_mean, scaler_scale, model, control_feats
        )
        
        # Compute overall metrics
        metrics = self.metrics.compute_all(
            y_true=y_test,
            y_scores=test_scores,
            control_threshold=control_threshold
        )
        
        # Compute per-dataset metrics
        if dataset_ids_test is not None and dataset_names is not None:
            grouped_metrics = self.metrics.compute_grouped(
                y_true=y_test,
                y_scores=test_scores,
                dataset_ids=dataset_ids_test,
                dataset_names=dataset_names,
                control_threshold=control_threshold
            )
        else:
            grouped_metrics = {'_overall': metrics}
        
        return {
            'mean': scaler_mean,
            'scale': scaler_scale,
            'directions': self._get_model_directions(model),
            'control_threshold': control_threshold,
            'train_accuracy': train_accuracy,
            'training_time': training_time,
            'metrics': metrics,
            'grouped_metrics': grouped_metrics,
            'use_scaler': self.use_scaler,
        }
    
    def _get_decision_scores(self, model, X: np.ndarray) -> np.ndarray:
        """Get decision scores from model."""
        if hasattr(model, 'decision_function'):
            return model.decision_function(X)
        # For models without decision_function, use predict_proba
        probs = model.predict_proba(X)
        return probs[:, 1] - probs[:, 0]
    
    def _get_model_directions(self, model) -> torch.Tensor | None:
        """Extract direction coefficients from model."""
        if hasattr(model, 'get_direction'):
            direction = model.get_direction()
            if direction is not None:
                return torch.tensor(np.array(direction)).squeeze()
        if hasattr(model, 'coef_'):
            return torch.tensor(model.coef_).squeeze()
        return None
    
    def _compute_control_threshold(
        self,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
        model,
        control_feats
    ) -> float:
        """Compute control threshold if applicable."""
        if not self.compute_control or control_feats is None:
            return np.nan
        if not hasattr(model, 'coef_'):
            return np.nan
        
        return utils.get_control_threshold(
            scaler_mean,
            scaler_scale,
            torch.tensor(model.coef_).squeeze(),
            control_feats,
        )
    
    def _print_cv_summary(self, aggregated: dict, aggregated_per_dataset: dict = None):
        """Print cross-validation summary including per-dataset results."""
        print("\n--- Overall CV Summary ---")
        for metric_name in self.metrics.metric_names:
            avg_key = f'avg_{metric_name}'
            std_key = f'std_{metric_name}'
            if avg_key in aggregated and not np.isnan(aggregated[avg_key]):
                display_name = f"CV average {metric_name}"
                avg_val = aggregated[avg_key]
                std_val = aggregated.get(std_key, np.nan)
                if np.isnan(std_val):
                    print(f"{display_name}: {avg_val:.4f}")
                else:
                    print(f"{display_name}: {avg_val:.4f} ± {std_val:.4f}")
        
        # Print per-dataset summary
        if aggregated_per_dataset and len(aggregated_per_dataset) > 1:
            print("\n--- Per-Dataset CV Summary ---")
            for dataset_name, ds_aggregated in aggregated_per_dataset.items():
                if dataset_name == '_overall':
                    continue
                
                # Get the primary metric (first non-control metric)
                primary_metric = None
                for m in self.metrics.metric_names:
                    if m != 'control_acc':
                        primary_metric = m
                        break
                
                if primary_metric:
                    avg_key = f'avg_{primary_metric}'
                    std_key = f'std_{primary_metric}'
                    avg_val = ds_aggregated.get(avg_key, np.nan)
                    std_val = ds_aggregated.get(std_key, np.nan)
                    
                    if not np.isnan(avg_val):
                        if np.isnan(std_val):
                            print(f"  {dataset_name}: {primary_metric}={avg_val:.4f}")
                        else:
                            print(f"  {dataset_name}: {primary_metric}={avg_val:.4f} ± {std_val:.4f}")
    
    def _create_metadata(
        self,
        task_name: str,
        layer_idx: int,
        train_feature_type: str,
        aggregated: dict,
        aggregated_per_dataset: dict = None,
        dataset_names: dict[int, str] = None
    ) -> dict:
        """Create metadata dictionary for saved probe."""
        metadata = {
            'task_name': task_name,
            'layer_idx': layer_idx,
            'train_feature_type': train_feature_type,
            'probe_type': self.probe_type,
            'regularization': self.regularization,
            'n_folds': self.n_folds,
            'use_scaler': self.use_scaler,
            'balance_groups': self.balance_groups,
            'metric_names': self.metrics.metric_names,
        }

        metadata.update(aggregated)
        
        # Add per-dataset metrics
        if aggregated_per_dataset:
            metadata['per_dataset_metrics'] = aggregated_per_dataset
            metadata['constituent_datasets'] = list(dataset_names.values()) if dataset_names else []
        
        return metadata
    
    def save_probe(self, cv_models: dict, filepath: str):
        """Save probe models to file."""
        torch.save(cv_models, filepath)
        if self.verbose:
            print(f"Saved probe to {filepath}")
    
    def get_probe_filename(
        self,
        layer_idx: int,
        probe_type: str = None,
        regularization: float = None
    ) -> str:
        """
        Generate the probe filename with use_scaler suffix.
        
        Args:
            layer_idx: Layer index
            probe_type: Probe type (uses self.probe_type if None)
            regularization: Regularization value (uses self.regularization if None)
        
        Returns:
            Filename string like 'probe_layer5_lr_1.0_scaled.pt'
        """
        probe_type = probe_type or self.probe_type
        regularization = regularization if regularization is not None else self.regularization
        scaler_suffix = "scaled" if self.use_scaler else "unscaled"
        balance_suffix = "balanced" if self.balance_groups else "unbalanced"
        return f"probe_layer{layer_idx}_{probe_type}_{regularization}_{scaler_suffix}_{balance_suffix}.pt"
    
    def train_probes_multi_layer(
        self,
        task_name: str,
        layer_idx_list: list[int],
        extracted_feats,
        control_feats,
        all_datasets,
        tokenizer,
        train_feature_type: str
    ) -> dict:
        """
        Train probes for multiple layers.
        
        Returns:
            Dictionary mapping layer indices to probe results
        """
        print(f"Training {self.probe_type} probes for {len(layer_idx_list)} layers")
        
        layer_results = {}
        
        for layer_idx in layer_idx_list:
            print(f"\n--- Layer {layer_idx} ---")
            
            results = self.train_probe_single_layer(
                task_name=task_name,
                layer_idx=layer_idx,
                extracted_feats=extracted_feats,
                control_feats=control_feats,
                all_datasets=all_datasets,
                tokenizer=tokenizer,
                train_feature_type=train_feature_type
            )
            
            if results is not None:
                layer_results[layer_idx] = results
        
        # Add overall metadata
        layer_results['metadata'] = {
            'task_name': task_name,
            'train_feature_type': train_feature_type,
            'probe_type': self.probe_type,
            'regularization': self.regularization,
            'num_layers': len(layer_idx_list),
            'use_scaler': self.use_scaler,
            'balance_groups': self.balance_groups,
            'metric_names': self.metrics.metric_names,
        }
        
        return layer_results