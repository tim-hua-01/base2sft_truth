import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import accuracy_score

from src.datasets import DialogueDataset
from src.tokenized_data import TokenizedDataset
from config import COMBINED_TASKS_CONFIG

#################################################
# TASK AND DATASET PARSING
#################################################

def parse_task_name(task_name):
    """
    Parse a task name into its components: dataset type and variant.
    
    Args:
        task_name (str): Full task name in format 'dataset__variant'
        
    Returns:
        tuple: (dataset_type, variant)
    """
    parts = task_name.split('__')
    base = parts[0]
    variant = parts[1] if len(parts) > 1 else None
    
    return base, variant

# def get_labels(task_name, all_datasets, label_type):
# this is just all_datasets[task_name].labels

def get_all_dataset(task_list, model_name):
    # Initialize the repository and model tokenizer
    all_datasets = {}
    for task in task_list:
        dataset = DialogueDataset(task, model_name)
        
        if 'alpaca' in task: # only 1000 samples for alpaca
            dataset = dataset[:1000]
            
        all_datasets[task] = dataset

    return all_datasets

#################################################
# FEATURE PROCESSING
#################################################

def get_feats_for_task(task_name, layer_name, extracted_feats):
    """
    Load features for a specific task.
    
    Args:
        task_name (str): Name of the task
        extracted_feats (dict): Dictionary of extracted features
        
    Returns:
        torch.Tensor: Features for the task
    """
    # Extract base task name (without label type)
    start_time = time.time()

    # Check if features exist for this task
    if layer_name not in extracted_feats:
        raise KeyError(f"Features not found for layer {layer_name}")
    if task_name not in extracted_feats[layer_name]:
        raise KeyError(f"Features not found for task {task_name} in layer {layer_name}")
    
    feats = torch.tensor(np.array(extracted_feats[layer_name][task_name][:]))
    
    print(f"loaded features after {time.time() - start_time:.4f}s")   
    return feats

def get_scores_for_feats(feats, mean, scaler, direction, max_chunk_size=20000):
    """
    Calculate scores for features using a learned direction with chunking support.
    
    Args:
        feats (torch.Tensor): Feature vectors
        mean (torch.Tensor): Mean vector for normalization
        scaler (torch.Tensor): Scaling vector for normalization
        direction (torch.Tensor): Direction vector for projection
        max_chunk_size (int): Maximum number of features to process at once
        
    Returns:
        torch.Tensor: Scores
    """
    start_time = time.time()    
    # Check if chunking is needed
    if feats.shape[0] <= max_chunk_size:
        # Process normally if small enough
        feats_norm = ((feats - mean) / scaler).cuda().to(torch.float16)
        scores = torch.matmul(feats_norm, direction.cuda().to(torch.float16)).squeeze()
        print(f"finished computing scores after {time.time()-start_time:.2f}s")
        return scores.cpu()
    
    # Process in chunks
    all_scores = []
    
    # Move direction to GPU once
    direction_gpu = direction.cuda().to(torch.float16)
    
    for i in range(0, feats.shape[0], max_chunk_size):
        # Get chunk
        end_idx = min(i + max_chunk_size, feats.shape[0])
        chunk = feats[i:end_idx]
        
        # Normalize chunk
        chunk_norm = ((chunk - mean) / scaler).cuda().to(torch.float16)
        
        # Project onto direction
        chunk_scores = torch.matmul(chunk_norm, direction_gpu).squeeze()
        
        # Move to CPU and store
        all_scores.append(chunk_scores.cpu())
        
        # Clear GPU memory for the chunk
        del chunk_norm, chunk_scores
        torch.cuda.empty_cache()
    
    # Concatenate all scores
    print(f"finished computing scores after {time.time()-start_time:.2f}")
    return torch.cat(all_scores, dim=0).cpu()

def get_detection_mask(dataset, tokenizer):
    """
    Get detection mask for a task.
    
    Args:
        task_name (str): Name of the task
        all_datasets (dict): Dictionary of all datasets
        tokenizer: Tokenizer for the model
        
    Returns:
        numpy.ndarray: Detection mask
    """
    tokenized_ds = TokenizedDataset.from_dataset(dataset, tokenizer)
    detection_mask = tokenized_ds.detection_mask
    return detection_mask

def extract_final_token_representations(X, sample_indices):
    """
    Extract final token representations for each sample.
    
    Args:
        X (numpy.ndarray): Token representations
        sample_indices (numpy.ndarray): Indices mapping tokens to samples
        
    Returns:
        numpy.ndarray: Final token representations for each sample
    """
    # Initialize the output array
    final_representations = np.zeros((sample_indices.max()+1, X.shape[1]))
    
    # Iterate through all tokens
    for i, sample_idx in enumerate(sample_indices):
        # Update the representation for this sample
        # By the end, we'll have the representation from the last occurrence
        final_representations[sample_idx] = X[i]
    
    return final_representations

#################################################
# DATA PREPARATION
#################################################

def group_based_on_mask(flat_list, detection_mask):
    """
    Group scores based on detection mask.
    
    Args:
        flat_list (list): Flat list of scores
        detection_mask (numpy.ndarray): Detection mask
        
    Returns:
        list: Grouped scores by sample
    """
    sample_token_mapping = []
    for sample_idx in range(detection_mask.shape[0]):
        for token_idx in range(detection_mask.shape[1]):
            if detection_mask[sample_idx, token_idx]:
                sample_token_mapping.append((sample_idx, token_idx))
    
    grouped_scores = [[] for _ in range(detection_mask.shape[0])]
    for flat_idx, (sample_idx, _) in enumerate(sample_token_mapping):
        if flat_idx < len(flat_list):
            grouped_scores[sample_idx].append(flat_list[flat_idx])
    return grouped_scores

@dataclass
class PreparedData:
    """Container for prepared training/evaluation data with sample origins."""
    X: np.ndarray
    y: np.ndarray
    dataset_ids: np.ndarray  # Integer ID for each sample
    dataset_names: dict[int, str]  # Mapping from ID to task name
    
    def __len__(self):
        return len(self.y)
    
    @property
    def n_datasets(self) -> int:
        return len(self.dataset_names)
    
    def get_dataset_mask(self, dataset_id: int) -> np.ndarray:
        """Get boolean mask for samples from a specific dataset."""
        return self.dataset_ids == dataset_id
    
    def get_dataset_indices(self, dataset_id: int) -> np.ndarray:
        """Get indices of samples from a specific dataset."""
        return np.where(self.dataset_ids == dataset_id)[0]


def prepare_sample_data(
    task_name: str,
    feats,
    detection_mask: np.ndarray,
    numerical_labels: np.ndarray,
    feature_type: str,
    balanced: bool = False,
    dataset_id: int = 0
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for training by grouping features by sample.
    
    Args:
        task_name: Name of the task
        feats: Feature vectors
        detection_mask: Detection mask
        numerical_labels: Numerical labels
        feature_type: Type of feature aggregation ('average', 'all', or 'last')
        balanced: Whether to balance positive and negative samples
        dataset_id: Integer ID to assign to samples from this dataset
        
    Returns:
        tuple: (X, y, dataset_ids) or (None, None, None) if no valid samples
    """
    assert feature_type in ['average', 'all', 'last']
    print(f"Preparing sample data for {task_name} (feature type={feature_type}, balanced={balanced})")
    
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(np.array(feats))
    
    # Create a mapping from sample index to token indices
    sample_to_tokens = {}
    flat_idx = 0
    for i in range(detection_mask.shape[0]):
        for j in range(detection_mask.shape[1]):
            if detection_mask[i, j]:
                if i not in sample_to_tokens:
                    sample_to_tokens[i] = []
                sample_to_tokens[i].append(flat_idx)
                flat_idx += 1
    
    X_list = []
    y_list = []
    
    for sample_idx, token_indices in sample_to_tokens.items():
        label = numerical_labels[sample_idx]
        if label not in [0, 1] or not token_indices:
            continue
        
        if feature_type == 'average':
            X_list.append(torch.mean(feats[token_indices], dim=0))
            y_list.append(label)
        elif feature_type == 'last':
            X_list.append(feats[token_indices[-1]])
            y_list.append(label)
        else:  # 'all'
            for token_idx in token_indices:
                X_list.append(feats[token_idx])
                y_list.append(label)
    
    if not X_list:
        print(f"  No valid samples found for {task_name}")
        return None, None, None
    
    X = torch.stack(X_list, dim=0).numpy()
    y = np.array(y_list)
    dataset_ids = np.full(len(y), dataset_id, dtype=np.int32)
    
    if X.ndim == 1:
        X = X[np.newaxis, :]

    if balanced:
        X, y, dataset_ids = _balance_samples(X, y, dataset_ids)
    
    print(f"  Prepared {len(y)} samples")
    return X, y, dataset_ids


def _balance_samples(
    X: np.ndarray,
    y: np.ndarray,
    dataset_ids: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Balance dataset by undersampling the majority class."""
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    n_samples = min(len(pos_indices), len(neg_indices))
    
    if n_samples == 0:
        print("  Warning: One class has no samples, cannot balance")
        return X, y, dataset_ids
    
    selected_pos = np.random.choice(pos_indices, n_samples, replace=False)
    selected_neg = np.random.choice(neg_indices, n_samples, replace=False)
    selected_indices = np.sort(np.concatenate([selected_pos, selected_neg]))
    
    result_ids = dataset_ids[selected_indices] if dataset_ids is not None else None
    return X[selected_indices], y[selected_indices], result_ids


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    dataset_ids: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Randomly subsample to n_samples."""
    n_to_take = min(n_samples, len(y))
    indices = np.random.choice(len(y), n_to_take, replace=False)
    result_ids = dataset_ids[indices] if dataset_ids is not None else None
    return X[indices], y[indices], result_ids


def prepare_data(
    task_name: str,
    layer_name: str,
    extracted_feats,
    all_datasets,
    tokenizer,
    balance_groups: bool,
    feature_type: str,
    random_seed: int = 42,
) -> PreparedData | tuple[None, None]:
    """
    Prepare data for training/evaluation with proper dataset balancing.
    
    Args:
        task_name: Name of the task
        layer_name: Name of the layer
        extracted_feats: Dictionary of extracted features
        all_datasets: Dictionary of all datasets
        tokenizer: Tokenizer for the model
        feature_type: Type of feature aggregation ('average', 'all', or 'last')
        random_seed: Random seed for reproducible sampling
        balance_groups: Whether to balance group sizes to the minimum (default: True)
        
    Returns:
        PreparedData object or (None, None) if no valid data
    """
    np.random.seed(random_seed)
    
    # Individual task handling
    if task_name not in COMBINED_TASKS_CONFIG:
        feats = get_feats_for_task(task_name, layer_name, extracted_feats)
        numerical_labels = all_datasets[task_name].labels
        detection_mask = get_detection_mask(all_datasets[task_name], tokenizer)
        
        X, y, dataset_ids = prepare_sample_data(
            task_name, feats, detection_mask, numerical_labels,
            feature_type=feature_type, balanced=False, dataset_id=0
        )
        
        if X is None:
            return None, None
        
        return PreparedData(
            X=X,
            y=y,
            dataset_ids=dataset_ids,
            dataset_names={0: task_name}
        )
    
    # Combined dataset handling
    config = COMBINED_TASKS_CONFIG[task_name]
    task_list = config['tasks']
    group_structure = config['groups']
    
    # Collect data from each dataset with unique IDs
    dataset_data = {}
    for task_id, task in enumerate(task_list):
        feats = get_feats_for_task(task, layer_name, extracted_feats)
        numerical_labels = all_datasets[task].labels
        detection_mask = get_detection_mask(all_datasets[task], tokenizer)
        
        X, y, dataset_ids = prepare_sample_data(
            task, feats, detection_mask, numerical_labels,
            feature_type=feature_type, balanced=False, dataset_id=task_id
        )
        
        if X is not None:
            dataset_data[task_id] = {
                'X': X,
                'y': y,
                'dataset_ids': dataset_ids,
                'task_name': task
            }
    
    if not dataset_data:
        print(f"  No valid data found for any dataset in {task_name}")
        return None, None
    
    # Calculate group sizes and find minimum
    group_sizes = [
        sum(len(dataset_data[tid]['y']) for tid in group if tid in dataset_data)
        for group in group_structure
    ]
    non_empty = [(g, s) for g, s in zip(group_structure, group_sizes) if s > 0]
    
    if not non_empty:
        return None, None
    
    min_group_size = min(s for _, s in non_empty)
    
    if balance_groups:
        print(f"Group sizes: {group_sizes}, balancing to {min_group_size}")
    else:
        print(f"Group sizes: {group_sizes}, using all data (no balancing)")
    
    # Collect data from each group
    all_X, all_y, all_dataset_ids = [], [], []
    dataset_names = {}
    
    for group, size in non_empty:
        valid_tasks = [tid for tid in group if tid in dataset_data]
        
        if balance_groups:
            samples_per_task = min_group_size // len(valid_tasks)
            remainder = min_group_size % len(valid_tasks)
        
        for i, tid in enumerate(valid_tasks):
            data = dataset_data[tid]
            
            if balance_groups:
                n_to_take = samples_per_task + (1 if i < remainder else 0)
                X_sampled, y_sampled, ids_sampled = _subsample(
                    data['X'], data['y'], n_to_take, data['dataset_ids']
                )
            else:
                X_sampled = data['X']
                y_sampled = data['y']
                ids_sampled = data['dataset_ids']
            
            print(f"  {data['task_name']}: {len(y_sampled)} samples")
            all_X.append(X_sampled)
            all_y.append(y_sampled)
            all_dataset_ids.append(ids_sampled)
            dataset_names[tid] = data['task_name']
    
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_dataset_ids = np.concatenate(all_dataset_ids, axis=0)
    
    print(f"Final combined dataset: {len(all_y)} entries from {len(dataset_names)} datasets")
    
    return PreparedData(
        X=all_X,
        y=all_y,
        dataset_ids=all_dataset_ids,
        dataset_names=dataset_names
    )

#################################################
# SCORING AND EVALUATION
#################################################

def compute_max_acc(avg_scores, numerical_labels):
    """
    Compute maximum accuracy by trying different thresholds.
    
    Args:
        avg_scores (numpy.ndarray): Average scores
        numerical_labels (numpy.ndarray): Numerical labels
        
    Returns:
        tuple: (max_accuracy, optimal_threshold)
    """
    unique_scores = np.unique(avg_scores)
    thresholds = np.concatenate([
        [unique_scores[0] - 0.001],
        (unique_scores[:-1] + unique_scores[1:]) / 2,
        [unique_scores[-1] + 0.001]
    ])
    
    # Calculate accuracy for each threshold
    max_accuracy = 0
    optimal_threshold = None
    
    for threshold in thresholds:
        predictions = (avg_scores >= threshold).astype(int)
        acc = accuracy_score(numerical_labels, predictions)
        
        if acc > max_accuracy:
            max_accuracy = acc
            optimal_threshold = threshold
    return max_accuracy, optimal_threshold

def compute_control_acc(avg_scores, numerical_labels, control_threshold):
    """
    Compute accuracy using a control threshold.
    
    Args:
        avg_scores (numpy.ndarray): Average scores
        numerical_labels (numpy.ndarray): Numerical labels
        control_threshold (float): Control threshold
        
    Returns:
        float: Control accuracy
    """
    predictions = (avg_scores >= control_threshold).astype(int)
    control_accuracy = accuracy_score(numerical_labels, predictions)
    return control_accuracy

def get_control_threshold(mean, scaler, directions, control_feats, fpr=0.01):
    """
    Calculate a control threshold based on features.
    
    Args:
        mean (torch.Tensor): Mean vector for normalization
        scaler (torch.Tensor): Scaling vector for normalization
        directions (torch.Tensor): Direction vector for projection
        control_feats (torch.Tensor): Control features
        fpr (float): False positive rate
        
    Returns:
        float: Control threshold
    """
    start_time = time.time()

    # Compute scores
    scores = get_scores_for_feats(control_feats, mean, scaler, directions)
    
    threshold = torch.quantile(scores, 1 - fpr)
    print(f"get control threshold after {time.time()-start_time}s")
    return threshold.cpu().numpy()