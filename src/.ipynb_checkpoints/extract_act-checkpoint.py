"""
Activation extraction using nnsight for Llama 3 models.

This module provides functionality to extract hidden state activations
from Llama 3 models using the nnsight library, which provides a clean
interface for accessing model internals.
"""

import torch
from torch import Tensor
from nnsight import LanguageModel
from tqdm import trange
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from src.tokenized_data import TokenizedDataset

@dataclass
class TokenizedBatch:
    """Represents a batch of tokenized inputs."""
    tokens: Tensor
    attention_mask: Tensor


@torch.inference_mode()
def extract_activation_nnsight(
    model: LanguageModel,
    tokenized_dataset: TokenizedDataset,
    save_logits: bool = False,
    batch_size: int = 16,
    layers: Optional[List[int]] = None,
    verbose: bool = False,
    detection_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Extract activations from a Llama 3 model using nnsight.
    
    This function uses nnsight's tracing context to efficiently extract
    hidden state activations from specified layers of the model.
    
    Args:
        model: An nnsight LanguageModel instance (Llama 3).
        tokenized_dataset: Dataset object with .tokens and .attention_mask attributes.
            Should support slicing to return batches.
        save_logits: If True, also extract and return the model's logits.
        batch_size: Number of samples to process in each forward pass.
        layers: List of layer indices to extract activations from.
            If None, extracts from all layers (including embeddings).
            Supports negative indices (e.g., -1 for last layer).
        verbose: If True, display a progress bar.
        detection_mask: Optional boolean tensor of shape [num_samples, seq_len].
            If provided, only activations at True positions are kept,
            saving memory by not storing full sequence activations.
    
    Returns:
        Tuple of (activations, logits):
            - If detection_mask is None:
                activations: Tensor of shape [batch, seq_len, num_layers, hidden_dim]
            - If detection_mask is provided:
                activations: Tensor of shape [total_masked_tokens, num_layers, hidden_dim]
            - logits: Tensor of shape [batch, seq_len, vocab_size] if save_logits=True,
                      else None (note: logits are NOT masked even if detection_mask is provided)
    """
    # Get model configuration
    num_hidden_layers = model.model.config.num_hidden_layers
    num_layers_total = num_hidden_layers + 1  # +1 for embeddings
    
    # Resolve layer indices (handle negative indices)
    if layers is not None:
        resolved_layers = [l % num_layers_total for l in layers]
    else:
        resolved_layers = list(range(num_layers_total))
    
    activations_by_batch = []
    logits_by_batch = []
    
    # Main processing loop
    iterator = trange(
        0, len(tokenized_dataset), batch_size,
        desc="Extracting activations with nnsight",
        disable=not verbose,
        unit="batch"
    )
    
    for i in iterator:
        # Get batch slice
        batch_slice = tokenized_dataset[i : i + batch_size]
        input_ids = batch_slice.tokens
        attention_mask = batch_slice.attention_mask
        
        # Use nnsight tracing context to extract activations
        saved_states = []
        with model.trace({"input_ids": input_ids, "attention_mask": attention_mask}) as tracer:
            # Collect hidden states from specified layers
            for layer_idx in resolved_layers:
                if layer_idx == 0:
                    # Layer 0 is the embedding layer output
                    state = model.model.embed_tokens.output.save()
                else:
                    # Transformer layers are 0-indexed in model.model.layers
                    # state = model.model.layers[layer_idx - 1].output[0].save()
                    state = model.model.layers[layer_idx - 1].output.save()
                saved_states.append(state.detach().cpu())
            
            # Optionally save logits
            if save_logits:
                batch_logits = model.lm_head.output.save()
        
        # Stack saved states: [num_layers, batch, seq_len, hidden_dim]
        stacked = torch.stack(saved_states, dim=0)
        # Transpose to: [batch, seq_len, num_layers, hidden_dim]
        stacked = stacked.permute(1, 2, 0, 3)
        
        # Apply detection mask if provided
        if detection_mask is not None:
            batch_mask = detection_mask[i : i + batch_size]  # [batch, seq_len]
            # Apply mask: result shape is [num_masked_tokens, num_layers, hidden_dim]
            stacked = stacked[batch_mask]
        
        activations_by_batch.append(stacked.cpu())
        
        if save_logits:
            logits_by_batch.append(batch_logits.value.cpu())
    
    # Concatenate all batches
    all_activations = torch.cat(activations_by_batch, dim=0)
    all_logits = torch.cat(logits_by_batch, dim=0) if save_logits else None
    
    return all_activations, all_logits