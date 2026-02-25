"""
Centralized metrics configuration and computation for probe training/evaluation.

To add a new metric:
1. Add computation function to METRIC_FUNCTIONS
2. Add entry to METRICS_CONFIG with desired settings
3. That's it - the metric will automatically be tracked in training, evaluation, and results
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from dataclasses import dataclass
from typing import Callable, Optional

from src import utils


@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    name: str                          # Short name used in code/columns (e.g., 'auroc')
    display_name: str                  # Human-readable name for printing
    compute_fn: Callable               # Function(y_true, y_scores) -> float
    requires_control: bool = False     # Whether this metric needs control threshold
    higher_is_better: bool = True      # For potential future use in model selection


def _compute_max_acc(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> tuple[float, float]:
    """Compute maximum accuracy with optimal threshold."""
    return utils.compute_max_acc(y_scores, y_true)


def _compute_auroc(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> float:
    """Compute Area Under ROC Curve."""
    return roc_auc_score(y_true, y_scores)


def _compute_auprc(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> float:
    """Compute Area Under Precision-Recall Curve."""
    return average_precision_score(y_true, y_scores)


def _compute_control_acc(y_true: np.ndarray, y_scores: np.ndarray,
                         control_threshold: float = np.nan, **kwargs) -> float:
    """Compute accuracy at control threshold."""
    if np.isnan(control_threshold):
        return np.nan
    return utils.compute_control_acc(y_scores, y_true, control_threshold)


def _compute_avg_abs_score(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> float:
    """Compute average absolute score (mean distance from decision boundary)."""
    return float(np.mean(np.abs(y_scores)))


# =============================================================================
# METRICS CONFIGURATION - Add new metrics here
# =============================================================================

METRICS_CONFIG = {
    'max_acc': MetricConfig(
        name='max_acc',
        display_name='Max Accuracy',
        compute_fn=_compute_max_acc,
        requires_control=False,
    ),
    'auroc': MetricConfig(
        name='auroc',
        display_name='AUROC',
        compute_fn=_compute_auroc,
        requires_control=False,
    ),
    'auprc': MetricConfig(
        name='auprc',
        display_name='AUPRC',
        compute_fn=_compute_auprc,
        requires_control=False,
    ),
    'control_acc': MetricConfig(
        name='control_acc',
        display_name='Control Accuracy',
        compute_fn=_compute_control_acc,
        requires_control=True,
    ),
    'avg_abs_score': MetricConfig(
        name='avg_abs_score',
        display_name='Avg Abs Score',
        compute_fn=_compute_avg_abs_score,
        requires_control=False,
    ),
}

# Default metrics to compute (can be overridden)
DEFAULT_METRICS = ['max_acc', 'auroc', 'auprc', 'control_acc', 'avg_abs_score']


class MetricsComputer:
    """Handles computation and aggregation of all metrics."""
    
    def __init__(self, metric_names: list[str] = None, compute_control: bool = True):
        """
        Initialize metrics computer.
        
        Args:
            metric_names: List of metric names to compute. If None, uses DEFAULT_METRICS.
            compute_control: Whether to compute control-dependent metrics.
        """
        self.metric_names = metric_names or DEFAULT_METRICS
        self.compute_control = compute_control
        
        # Filter out control metrics if not computing them
        if not compute_control:
            self.metric_names = [
                m for m in self.metric_names 
                if not METRICS_CONFIG[m].requires_control
            ]
        
        self._validate_metrics()
    
    def _validate_metrics(self):
        """Validate that all requested metrics exist."""
        for name in self.metric_names:
            if name not in METRICS_CONFIG:
                raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS_CONFIG.keys())}")
    
    def compute_all(self, y_true: np.ndarray, y_scores: np.ndarray,
                    control_threshold: float = np.nan) -> dict[str, float]:
        """
        Compute all configured metrics.
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores
            control_threshold: Threshold for control accuracy
            
        Returns:
            Dictionary mapping metric names to values
        """
        results = {}
        extra_results = {}  # For things like optimal_threshold
        
        for name in self.metric_names:
            config = METRICS_CONFIG[name]
            
            try:
                result = config.compute_fn(
                    y_true=y_true,
                    y_scores=y_scores,
                    control_threshold=control_threshold
                )
                
                # Handle metrics that return tuples (like max_acc with threshold)
                if isinstance(result, tuple):
                    results[name] = result[0]
                    if name == 'max_acc':
                        extra_results['optimal_threshold'] = result[1]
                else:
                    results[name] = result
                    
            except Exception as e:
                results[name] = np.nan
        
        results.update(extra_results)
        return results
    
    def aggregate_folds(self, fold_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Aggregate metrics across folds.
        
        Args:
            fold_metrics: Dict mapping fold_name -> {metric_name: value}
            
        Returns:
            Dictionary with avg_* and std_* for each metric
        """
        aggregated = {}
        
        for metric_name in self.metric_names:
            values = []
            for fold_name, metrics in fold_metrics.items():
                value = metrics.get(metric_name, np.nan)
                if not np.isnan(value):
                    values.append(value)
            
            aggregated[f'avg_{metric_name}'] = np.mean(values) if values else np.nan
            aggregated[f'std_{metric_name}'] = np.std(values) if len(values) > 1 else np.nan
        
        return aggregated
    
    def get_results_columns(self, n_folds: int = 5) -> list[str]:
        """
        Get column names for results dataframe.
        
        Args:
            n_folds: Number of cross-validation folds
            
        Returns:
            List of column names
        """
        # Base columns
        columns = [
            'train_dataset', 'test_dataset', 'train_feature_type', 'test_feature_type',
            'model_name', 'layer_idx', 'probe_type', 'regularization', 'use_scaler', 'balance_groups', 'is_cv_result',
        ]
        
        # Aggregated metric columns
        for metric_name in self.metric_names:
            columns.extend([f'avg_{metric_name}', f'std_{metric_name}'])
        
        # Per-fold columns
        for i in range(n_folds):
            for metric_name in self.metric_names:
                columns.append(f'fold_{i+1}_{metric_name}')
        
        return columns
    
    def format_metrics_string(self, metrics: dict[str, float], prefix: str = "") -> str:
        """Format metrics for verbose printing."""
        parts = []
        for name in self.metric_names:
            if name in metrics and not np.isnan(metrics[name]):
                display_name = METRICS_CONFIG[name].display_name
                parts.append(f"{display_name}={metrics[name]:.4f}")
        return f"{prefix}{', '.join(parts)}"
    
    def compute_grouped(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        dataset_ids: np.ndarray,
        dataset_names: dict[int, str],
        control_threshold: float = np.nan
    ) -> dict[str, dict[str, float]]:
        """
        Compute metrics grouped by dataset.
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores
            dataset_ids: Dataset ID for each sample
            dataset_names: Mapping from dataset ID to name
            control_threshold: Threshold for control accuracy
            
        Returns:
            Dictionary mapping dataset names to their metrics
        """
        grouped_results = {}
        
        # Compute overall metrics
        grouped_results['_overall'] = self.compute_all(
            y_true=y_true,
            y_scores=y_scores,
            control_threshold=control_threshold
        )
        
        # Compute per-dataset metrics
        unique_ids = np.unique(dataset_ids)
        for dataset_id in unique_ids:
            mask = dataset_ids == dataset_id
            
            if np.sum(mask) < 2:  # Need at least 2 samples
                continue
            
            dataset_name = dataset_names.get(dataset_id, f"dataset_{dataset_id}")
            
            try:
                dataset_metrics = self.compute_all(
                    y_true=y_true[mask],
                    y_scores=y_scores[mask],
                    control_threshold=control_threshold
                )
                grouped_results[dataset_name] = dataset_metrics
            except Exception as e:
                # If computation fails for a dataset, store NaNs
                grouped_results[dataset_name] = {name: np.nan for name in self.metric_names}
        
        return grouped_results
    
    def aggregate_grouped_folds(
        self,
        fold_grouped_metrics: dict[str, dict[str, dict[str, float]]]
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate grouped metrics across folds.
        
        Args:
            fold_grouped_metrics: Dict mapping fold_name -> dataset_name -> metrics
            
        Returns:
            Dictionary mapping dataset_name -> aggregated metrics (avg_*, std_*)
        """
        # Collect all dataset names across folds
        all_dataset_names = set()
        for fold_metrics in fold_grouped_metrics.values():
            all_dataset_names.update(fold_metrics.keys())
        
        aggregated = {}
        
        for dataset_name in all_dataset_names:
            # Collect metrics for this dataset across all folds
            dataset_fold_metrics = {}
            for fold_name, fold_data in fold_grouped_metrics.items():
                if dataset_name in fold_data:
                    dataset_fold_metrics[fold_name] = fold_data[dataset_name]
            
            # Use standard fold aggregation
            aggregated[dataset_name] = self.aggregate_folds(dataset_fold_metrics)
        
        return aggregated
    
    def get_per_dataset_columns(self, dataset_names: list[str], n_folds: int = 5) -> list[str]:
        """
        Get column names for per-dataset results.
        
        Args:
            dataset_names: List of dataset names
            n_folds: Number of folds
            
        Returns:
            List of column names for per-dataset metrics
        """
        columns = []
        
        for dataset_name in dataset_names:
            # Sanitize dataset name for column use
            safe_name = dataset_name.replace('__', '_').replace('-', '_')
            
            for metric_name in self.metric_names:
                columns.append(f'{safe_name}_avg_{metric_name}')
                columns.append(f'{safe_name}_std_{metric_name}')
        
        return columns