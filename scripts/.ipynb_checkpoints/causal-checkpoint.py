"""
Causal Assessment of Truth Directions via Bias Modification

Measures the causal effect of general vs domain-specific truth directions
by adding them to layer bias terms and measuring change in P(correct answer).

For SimpleQA: Creates incorrect answers by sampling other entities of the same type,
then measures the change in likelihood ratio P(correct)/P(incorrect).

For Claims: Uses true/false claim pairs.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import json
import ast
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from contextlib import contextmanager
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures


# ============================================================================
# Hardcoded True/False Claim Pairs (35 pairs)
# ============================================================================

CLAIM_PAIRS = [
    {"true": "A triangle is a polygon with three sides.", "false": "A triangle is a polygon with four sides."},
    {"true": "Water is a molecule consisting of two hydrogen atoms and one oxygen atom.", "false": "Water is a molecule consisting of three hydrogen atoms and two carbon atoms."},
    {"true": "A bachelor is an unmarried man.", "false": "A bachelor is a married man."},
    {"true": "An ophthalmologist is a medical doctor who specializes in eye and vision care.", "false": "An ophthalmologist is a medical doctor who specializes in foot and ankle surgery."},
    {"true": "An octagon is a geometric shape with eight sides.", "false": "An octagon is a geometric shape with three sides."},
    {"true": "A century is a period of one hundred years.", "false": "A century is a period of five hundred years."},
    {"true": "Helium is a chemical element with the atomic number two.", "false": "Helium is a chemical element with the atomic number ninety-two."},
    {"true": "A sibling is a person who shares the same parents as another.", "false": "A sibling is a person who shares no genetic relation with another."},
    {"true": "Botany is the scientific study of plants.", "false": "Botany is the scientific study of the atmosphere and weather."},
    {"true": "A quadruped is an animal that has four feet.", "false": "A quadruped is an animal that has zero feet."},
    {"true": "Gold is a chemical element with the symbol Au and atomic number 79.", "false": "Gold is a chemical element with the symbol Ag and atomic number 47."},
    {"true": "A decade is a period of ten years.", "false": "A decade is a period of one hundred years."},
    {"true": "A prime number is a natural number greater than one that has no positive divisors other than one and itself.", "false": "A prime number is a natural number less than one that has many positive divisors."},
    {"true": "Horticulture is the art or practice of garden cultivation and management.", "false": "Horticulture is the study of oceanic currents and marine life."},
    {"true": "A sphere is a perfectly round geometrical object in three-dimensional space.", "false": "A sphere is a perfectly flat geometrical object in two-dimensional space."},
    {"true": "Biology is the natural science that studies life and living organisms.", "false": "Biology is the physical science that studies inorganic compounds and minerals."},
    {"true": "An orphan is a child whose parents are deceased.", "false": "An orphan is a child whose parents are both currently alive and healthy."},
    {"true": "A kilometer is a unit of length equal to one thousand meters.", "false": "A kilometer is a unit of length equal to five hundred meters."},
    {"true": "An isotope is an atom with the same number of protons but a different number of neutrons.", "false": "An isotope is an atom with a different number of protons but the same number of neutrons."},
    {"true": "Anthropology is the study of humans, human behavior, and societies.", "false": "Anthropology is the study of deep-sea volcanic eruptions."},
    {"true": "A hexagon is a polygon with six sides.", "false": "A hexagon is a polygon with ten sides."},
    {"true": "Mammals are vertebrate animals characterized by the presence of mammary glands.", "false": "Mammals are vertebrate animals characterized by the absence of mammary glands."},
    {"true": "A widow is a woman whose spouse has died and who has not remarried.", "false": "A widow is a woman who has never been married."},
    {"true": "Oxygen is a chemical element with the atomic number eight.", "false": "Oxygen is a chemical element with the atomic number one."},
    {"true": "A right angle is an angle of exactly ninety degrees.", "false": "A right angle is an angle of exactly forty-five degrees."},
    {"true": "Entomology is the scientific study of insects.", "false": "Entomology is the scientific study of stars and galaxies."},
    {"true": "A fortnight is a period of two weeks.", "false": "A fortnight is a period of six months."},
    {"true": "Geometry is a branch of mathematics concerned with the properties and relations of points, lines, and surfaces.", "false": "Geometry is a branch of biology concerned with the evolution of cell walls."},
    {"true": "A pentagon is a polygon with five sides.", "false": "A pentagon is a polygon with twelve sides."},
    {"true": "Geology is the science that deals with the earth's physical structure and substance.", "false": "Geology is the science that deals with the human subconscious and dreams."},
    {"true": "A year is the time taken by the earth to make one revolution around the sun.", "false": "A year is the time taken by the moon to revolve around the earth five times."},
    {"true": "An autotroph is an organism that produces complex organic compounds from simple substances.", "false": "An autotroph is an organism that must consume other organisms for energy."},
    {"true": "A synonym is a word that means the same or nearly the same as another word.", "false": "A synonym is a word that means the complete opposite of another word."},
    {"true": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.", "false": "Photosynthesis is the process by which animals generate oxygen by eating green plants."},
]


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ClaimInterventionResult:
    """Result of intervention on a true/false claim pair."""
    true_claim: str
    false_claim: str
    baseline_true_logprob: float
    baseline_false_logprob: float
    baseline_ratio: float
    baseline_diff: float
    intervened_true_logprob: float
    intervened_false_logprob: float
    intervened_ratio: float
    intervened_diff: float
    ratio_change: float
    diff_change: float
    direction_name: str
    alpha: float
    was_intervened: bool = True


@dataclass
class ClaimDirectionAssessment:
    """Aggregated assessment of one direction on claim pairs."""
    direction_name: str
    direction_type: str
    source_domain: Optional[str]
    alpha: float
    mean_baseline_diff: float
    mean_intervened_diff: float
    mean_diff_change: float
    std_diff_change: float
    median_diff_change: float
    mean_ratio_change: float
    median_ratio_change: float
    p_value: float
    n_intervened: int = 0
    n_skipped: int = 0
    results: List[ClaimInterventionResult] = field(default_factory=list)


@dataclass
class QAInterventionResult:
    """Result of intervention on a QA pair with correct/incorrect answers."""
    question: str
    correct_answer: str
    incorrect_answer: str
    answer_type: str
    baseline_correct_logprob: float
    baseline_incorrect_logprob: float
    baseline_diff: float
    intervened_correct_logprob: float
    intervened_incorrect_logprob: float
    intervened_diff: float
    diff_change: float
    direction_name: str
    alpha: float
    was_intervened: bool = True


@dataclass
class QADirectionAssessment:
    """Aggregated assessment of one direction on QA pairs."""
    direction_name: str
    direction_type: str
    source_domain: Optional[str]
    alpha: float
    n_examples: int
    mean_baseline_diff: float
    std_baseline_diff: float
    mean_intervened_diff: float
    mean_diff_change: float
    std_diff_change: float
    median_diff_change: float
    mean_correct_change: float
    mean_incorrect_change: float
    p_value: float
    n_intervened: int = 0
    n_skipped: int = 0
    results: List[QAInterventionResult] = field(default_factory=list)


# ============================================================================
# Bias Modification
# ============================================================================

def get_layer_bias_param(
    model: AutoModelForCausalLM,
    layer_idx: int,
    model_type: str = "llama",
    bias_location: str = "mlp_out",
) -> Tuple[Optional[torch.nn.Parameter], str]:
    """Get the bias parameter for a specific layer."""
    if model_type in ["llama", "mistral", "qwen2"]:
        layer = model.model.layers[layer_idx]
        if bias_location == "mlp_out":
            module = layer.mlp.down_proj
            path = f"model.layers.{layer_idx}.mlp.down_proj"
        elif bias_location == "attn_out":
            module = layer.self_attn.o_proj
            path = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            raise ValueError(f"Unknown bias_location: {bias_location}")
        if hasattr(module, 'bias') and module.bias is not None:
            return module.bias, path
        else:
            return None, path
    elif model_type == "gpt2":
        layer = model.transformer.h[layer_idx]
        if bias_location == "mlp_out":
            module = layer.mlp.c_proj
            path = f"transformer.h.{layer_idx}.mlp.c_proj"
        elif bias_location == "attn_out":
            module = layer.attn.c_proj
            path = f"transformer.h.{layer_idx}.attn.c_proj"
        else:
            raise ValueError(f"Unknown bias_location: {bias_location}")
        if hasattr(module, 'bias') and module.bias is not None:
            return module.bias, path
        else:
            return None, path
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def add_bias_to_layer(
    model: AutoModelForCausalLM,
    layer_idx: int,
    model_type: str = "llama",
    bias_location: str = "mlp_out",
) -> torch.nn.Parameter:
    """Add a bias parameter to a layer that doesn't have one."""
    if model_type in ["llama", "mistral", "qwen2"]:
        layer = model.model.layers[layer_idx]
        if bias_location == "mlp_out":
            module = layer.mlp.down_proj
        elif bias_location == "attn_out":
            module = layer.self_attn.o_proj
        else:
            raise ValueError(f"Cannot add bias to {bias_location}")
        hidden_size = module.out_features
        bias = torch.nn.Parameter(
            torch.zeros(hidden_size, dtype=module.weight.dtype, device=module.weight.device)
        )
        module.bias = bias
        return bias
    elif model_type == "gpt2":
        layer = model.transformer.h[layer_idx]
        if bias_location == "mlp_out":
            module = layer.mlp.c_proj
        elif bias_location == "attn_out":
            module = layer.attn.c_proj
        else:
            raise ValueError(f"Cannot add bias to {bias_location}")
        hidden_size = module.out_features if hasattr(module, 'out_features') else module.nf
        bias = torch.nn.Parameter(
            torch.zeros(hidden_size, dtype=module.weight.dtype, device=module.weight.device)
        )
        module.bias = bias
        return bias
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@contextmanager
def bias_intervention(
    model: AutoModelForCausalLM,
    layer_idx: int,
    direction: np.ndarray,
    alpha: float = 1.0,
    model_type: str = "llama",
    bias_location: str = "mlp_out",
    verbose: bool = True,
):
    """Context manager that temporarily modifies the bias of a layer."""
    direction_tensor = torch.tensor(direction, dtype=model.dtype, device=model.device)
    bias_param, path = get_layer_bias_param(model, layer_idx, model_type, bias_location)
    
    created_bias = False
    if bias_param is None:
        if verbose:
            print(f"  No bias at {path}, creating one...")
        bias_param = add_bias_to_layer(model, layer_idx, model_type, bias_location)
        created_bias = True
    
    if bias_param.shape[0] != direction_tensor.shape[0]:
        raise ValueError(f"Direction dim ({direction_tensor.shape[0]}) != bias dim ({bias_param.shape[0]})")
    
    original_bias = bias_param.data.clone()
    
    try:
        with torch.no_grad():
            bias_param.data = original_bias + alpha * direction_tensor
        yield bias_param
    finally:
        with torch.no_grad():
            if created_bias:
                if model_type in ["llama", "mistral", "qwen2"]:
                    layer = model.model.layers[layer_idx]
                    if bias_location == "mlp_out":
                        layer.mlp.down_proj.bias = None
                    elif bias_location == "attn_out":
                        layer.self_attn.o_proj.bias = None
                elif model_type == "gpt2":
                    layer = model.transformer.h[layer_idx]
                    if bias_location == "mlp_out":
                        layer.mlp.c_proj.bias = None
                    elif bias_location == "attn_out":
                        layer.attn.c_proj.bias = None
            else:
                bias_param.data = original_bias


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_sentence_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    device: str = "cuda",
) -> float:
    """Compute the average log-probability of a sentence."""
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor([token_ids], device=device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    total_logprob = 0.0
    n_tokens = 0
    
    for i in range(0, len(token_ids) - 1):
        next_token = token_ids[i + 1]
        token_logprob = log_probs[0, i, next_token].item()
        total_logprob += token_logprob
        n_tokens += 1
    
    if n_tokens == 0:
        return 0.0
    return total_logprob / n_tokens


def compute_answer_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer: str,
    device: str = "cuda",
) -> float:
    """Compute average log-probability of the answer given the prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    
    full_ids = prompt_ids + answer_ids
    full_ids_tensor = torch.tensor([full_ids], device=device)
    
    prompt_len = len(prompt_ids)
    
    with torch.no_grad():
        outputs = model(full_ids_tensor)
        logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    total_logprob = 0.0
    n_answer_tokens = 0
    
    for i in range(prompt_len - 1, len(full_ids) - 1):
        target_pos = i + 1
        target_token = full_ids[target_pos]
        token_logprob = log_probs[0, i, target_token].item()
        total_logprob += token_logprob
        n_answer_tokens += 1
    
    if n_answer_tokens == 0:
        return 0.0
    return total_logprob / n_answer_tokens


# ============================================================================
# Dataset Loading
# ============================================================================

def load_claim_pairs() -> List[Dict[str, str]]:
    """Load the hardcoded claim pairs."""
    return CLAIM_PAIRS.copy()


def load_simpleqa_with_distractors(
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load SimpleQA dataset and create incorrect answers by sampling other answers of the same answer_type."""
    import json as json_module
    
    try:
        dataset = load_dataset("basicv8vc/SimpleQA", split="test")
    except:
        try:
            dataset = load_dataset("openai/simple_qa", split="test")
        except:
            print("Could not load SimpleQA, using fallback examples...")
            return _fallback_qa_examples()
    
    answers_by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    all_examples = []
    
    for item in dataset:
        answer_type = None
        metadata = item.get("metadata", None)
        
        if metadata is not None:
            if isinstance(metadata, str):
                try:
                    metadata_dict = ast.literal_eval(metadata)
                    answer_type = metadata_dict.get("answer_type") or metadata_dict.get("topic")
                except (ValueError, SyntaxError):
                    try:
                        metadata_dict = json_module.loads(metadata)
                        answer_type = metadata_dict.get("answer_type") or metadata_dict.get("topic")
                    except (json_module.JSONDecodeError, TypeError):
                        pass
            elif isinstance(metadata, dict):
                answer_type = metadata.get("answer_type") or metadata.get("topic")
        
        if answer_type is None:
            answer_type = item.get("answer_type") or item.get("topic") or item.get("category") or "unknown"
        
        answer = item.get("answer", "")
        question = item.get("problem", item.get("question", ""))
        
        all_examples.append({
            "question": question,
            "correct_answer": answer,
            "answer_type": answer_type,
        })
        answers_by_type[answer_type].append((answer, question))
    
    print(f"\nAnswer types found ({len(answers_by_type)} types):")
    for atype, answers in sorted(answers_by_type.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"  {atype}: {len(answers)}")
    
    np.random.seed(seed)
    examples_with_distractors = []
    same_type_count = 0
    fallback_count = 0
    
    for ex in all_examples:
        answer_type = ex["answer_type"]
        correct_answer = ex["correct_answer"]
        question = ex["question"]
        
        same_type_answers = [a for a, q in answers_by_type[answer_type] 
                            if a != correct_answer and q != question]
        
        if len(same_type_answers) >= 1:
            incorrect_answer = np.random.choice(same_type_answers)
            same_type_count += 1
        else:
            all_other_answers = []
            for atype, answers_list in answers_by_type.items():
                if atype != answer_type:
                    all_other_answers.extend([a for a, q in answers_list if a != correct_answer])
            
            if len(all_other_answers) == 0:
                continue
            
            incorrect_answer = np.random.choice(all_other_answers)
            fallback_count += 1
        
        examples_with_distractors.append({
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
            "answer_type": answer_type,
        })
    
    print(f"\nSampling stats: {same_type_count} same-type, {fallback_count} fallback")
    
    if n_samples is not None and n_samples < len(examples_with_distractors):
        indices = np.random.choice(len(examples_with_distractors), n_samples, replace=False)
        examples_with_distractors = [examples_with_distractors[i] for i in indices]
    
    print("\nSample verification (first 3):")
    for i, ex in enumerate(examples_with_distractors[:3]):
        print(f"  {i+1}. Type: {ex['answer_type']}")
        print(f"      Q: {ex['question'][:60]}...")
        print(f"      ✓ Correct: {ex['correct_answer'][:40]}")
        print(f"      ✗ Incorrect: {ex['incorrect_answer'][:40]}")
    
    return examples_with_distractors


def _fallback_qa_examples() -> List[Dict[str, str]]:
    """Fallback QA examples with manually created distractors."""
    return [
        {"question": "What is the capital of France?", "correct_answer": "Paris", "incorrect_answer": "London", "answer_type": "Location"},
        {"question": "Who wrote Romeo and Juliet?", "correct_answer": "William Shakespeare", "incorrect_answer": "Charles Dickens", "answer_type": "Person"},
        {"question": "What is the chemical symbol for gold?", "correct_answer": "Au", "incorrect_answer": "Ag", "answer_type": "Other"},
        {"question": "What planet is closest to the Sun?", "correct_answer": "Mercury", "incorrect_answer": "Venus", "answer_type": "Location"},
        {"question": "Who painted the Mona Lisa?", "correct_answer": "Leonardo da Vinci", "incorrect_answer": "Michelangelo", "answer_type": "Person"},
    ]


def format_qa_prompt(question: str, prompt_template: Optional[str] = None) -> str:
    """Format a QA prompt."""
    if prompt_template is None:
        prompt_template = "Question: {question}\nAnswer:"
    return prompt_template.format(question=question)


# ============================================================================
# Visualization Helper Functions
# ============================================================================

def shorten_direction_name(name: str) -> str:
    """Shorten direction name for display."""
    name = name.replace('specific_', '').replace('internal_state__', 'int_')
    name = name.replace('claims__', 'cl_').replace('_gemini_600_full', '')
    return name


def compute_percentile_bins(data: List[float], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute bin edges based on percentiles and return edges, percentiles, and labels.
    Returns: (bin_edges, percentiles, bin_labels)
    """
    percentiles_arr = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data, percentiles_arr)
    bin_edges_unique = np.unique(bin_edges)
    
    if len(bin_edges_unique) < len(bin_edges):
        bin_edges = bin_edges_unique
        n_bins = len(bin_edges) - 1
        percentiles = []
        for edge in bin_edges:
            pct = 100 * np.mean(np.array(data) <= edge)
            percentiles.append(pct)
        percentiles = np.array(percentiles)
    else:
        percentiles = percentiles_arr
    
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        low_pct = percentiles[i]
        high_pct = percentiles[i + 1]
        bin_labels.append(f'{low_pct:.0f}-{high_pct:.0f}%')
    
    return bin_edges, percentiles, bin_labels


def bin_results_by_baseline(
    results: List[QAInterventionResult],
    bin_edges: np.ndarray,
    intervened_only: bool = True,
) -> List[List[QAInterventionResult]]:
    """Bin results by their baseline_diff values."""
    n_bins = len(bin_edges) - 1
    binned = [[] for _ in range(n_bins)]
    
    for r in results:
        if intervened_only and not r.was_intervened:
            continue
        
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                if low <= r.baseline_diff <= high:
                    binned[i].append(r)
                    break
            else:
                if low <= r.baseline_diff < high:
                    binned[i].append(r)
                    break
    
    return binned


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_direction_comparison(
    df: pd.DataFrame,
    output_path: Path,
    alpha: float,
    metric: str = "mean_diff_change",
    title_prefix: str = "",
):
    """Create a bar plot comparing diff_change across all directions WITH ERROR BARS."""
    df_alpha = df[df["alpha"] == alpha].copy()
    df_alpha = df_alpha.sort_values(metric, ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_alpha) * 0.4)))
    
    y_pos = np.arange(len(df_alpha))
    values = df_alpha[metric].values
    errors = df_alpha['std_diff_change'].values
    
    colors = []
    for val, dtype in zip(values, df_alpha['direction_type']):
        if val < 0:
            colors.append('#e74c3c')
        elif dtype == 'general':
            colors.append('#9b59b6')
        else:
            colors.append('#2ecc71')
    
    bars = ax.barh(y_pos, values, xerr=errors, color=colors, 
                   edgecolor='black', linewidth=0.5, capsize=3,
                   error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 0.7})
    
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.set_yticks(y_pos)
    
    short_names = [shorten_direction_name(name) for name in df_alpha['direction_name']]
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel(f'{metric} ± std (positive = better discrimination)', fontsize=11)
    ax.set_title(f'{title_prefix}Causal Effect by Direction (α={alpha})', fontsize=13)
    
    for i, (val, p, err) in enumerate(zip(values, df_alpha['p_value'], errors)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        label_x = val + err + 0.01 if val >= 0 else val - err - 0.01
        ax.text(label_x, i, f'{val:+.3f}{sig}', va='center', 
                ha='left' if val >= 0 else 'right', fontsize=8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', label='General'),
        Patch(facecolor='#2ecc71', label='Specific (positive)'),
        Patch(facecolor='#e74c3c', label='Negative effect'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / f'direction_comparison_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'direction_comparison_alpha{alpha}.png'}")


def plot_per_example_effects(
    results: List[QAInterventionResult],
    direction_name: str,
    output_path: Path,
    alpha: float,
):
    """Create scatter and histogram plots for a single direction."""
    baseline_diffs = [r.baseline_diff for r in results]
    intervened_diffs = [r.intervened_diff for r in results]
    diff_changes = [r.diff_change for r in results]
    was_intervened = [r.was_intervened for r in results]
    
    short_name = shorten_direction_name(direction_name)
    clean_name = direction_name.replace('[', '_').replace(']', '').replace('/', '_')
    
    n_intervened = sum(was_intervened)
    n_improved = sum(1 for dc, wi in zip(diff_changes, was_intervened) if dc > 0 and wi)
    n_hurt = sum(1 for dc, wi in zip(diff_changes, was_intervened) if dc < 0 and wi)
    
    # Plot 1: Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = []
    for dc, wi in zip(diff_changes, was_intervened):
        if not wi:
            colors.append('#aaaaaa')
        elif dc > 0:
            colors.append('#2ecc71')
        else:
            colors.append('#e74c3c')
    
    ax.scatter(baseline_diffs, intervened_diffs, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    min_val = min(min(baseline_diffs), min(intervened_diffs))
    max_val = max(max(baseline_diffs), max(intervened_diffs))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No change')
    
    ax.set_xlabel('Baseline diff (LogP(correct) - LogP(incorrect))', fontsize=11)
    ax.set_ylabel('Intervened diff', fontsize=11)
    ax.set_title(f'{short_name}: Baseline vs Intervened (α={alpha})', fontsize=12)
    ax.legend()
    
    ax.text(0.05, 0.95, f'Intervened: {n_intervened}/{len(results)}\nImproved: {n_improved}\nHurt: {n_hurt}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / f'effects_{clean_name}_scatter_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    intervened_diff_changes = [dc for dc, wi in zip(diff_changes, was_intervened) if wi]
    if intervened_diff_changes:
        ax.hist(intervened_diff_changes, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='No change')
        ax.axvline(x=np.mean(intervened_diff_changes), color='green', linewidth=2, linestyle='-', 
                   label=f'Mean: {np.mean(intervened_diff_changes):+.3f}')
    
    ax.set_xlabel('diff_change (positive = improved)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{short_name}: Distribution of Effects (α={alpha})', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / f'effects_{clean_name}_histogram_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_correct_vs_incorrect_changes(
    df: pd.DataFrame,
    output_path: Path,
    alpha: float,
):
    """Create a scatter plot showing correct_change vs incorrect_change for each direction."""
    df_alpha = df[df["alpha"] == alpha].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for _, row in df_alpha.iterrows():
        color = '#9b59b6' if row['direction_type'] == 'general' else '#2ecc71'
        if row['mean_diff_change'] < 0:
            color = '#e74c3c'
        marker = 'o' if row['direction_type'] == 'general' else 's'
        
        ax.scatter(row['mean_incorrect_change'], row['mean_correct_change'], 
                   c=color, marker=marker, s=100, edgecolors='black', linewidth=0.5)
        
        short_name = shorten_direction_name(row['direction_name'])[:15]
        ax.annotate(short_name, (row['mean_incorrect_change'], row['mean_correct_change']),
                    fontsize=7, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Equal change')
    
    ax.set_xlabel('Mean change in LogP(incorrect)', fontsize=11)
    ax.set_ylabel('Mean change in LogP(correct)', fontsize=11)
    ax.set_title(f'Correct vs Incorrect Log-Probability Changes (α={alpha})\n'
                 f'Above diagonal = diff_change > 0 (good)', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / f'correct_vs_incorrect_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'correct_vs_incorrect_alpha{alpha}.png'}")


def plot_effect_decomposition(
    df: pd.DataFrame,
    output_path: Path,
    alpha: float,
):
    """Create plots showing the decomposition of the effect into correct vs incorrect changes."""
    df_alpha = df[df["alpha"] == alpha].copy()
    df_alpha = df_alpha.sort_values('mean_diff_change', ascending=True)
    
    y_pos = np.arange(len(df_alpha))
    short_names = [shorten_direction_name(name) for name in df_alpha['direction_name']]
    
    width = 0.35
    correct_changes = df_alpha['mean_correct_change'].values
    incorrect_changes = df_alpha['mean_incorrect_change'].values
    
    # Plot 1: Grouped bars
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_alpha) * 0.4)))
    
    ax.barh(y_pos - width/2, correct_changes, width, label='ΔLogP(correct)', 
            color='#3498db', edgecolor='black', linewidth=0.5)
    ax.barh(y_pos + width/2, incorrect_changes, width, label='ΔLogP(incorrect)', 
            color='#e67e22', edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Change in Log-Probability', fontsize=11)
    ax.set_title(f'Decomposition: Correct vs Incorrect Changes (α={alpha})', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    
    for i, (cc, ic) in enumerate(zip(correct_changes, incorrect_changes)):
        ax.text(cc + 0.005 if cc >= 0 else cc - 0.005, i - width/2, 
                f'{cc:+.3f}', va='center', ha='left' if cc >= 0 else 'right', fontsize=7, color='#2980b9')
        ax.text(ic + 0.005 if ic >= 0 else ic - 0.005, i + width/2,
                f'{ic:+.3f}', va='center', ha='left' if ic >= 0 else 'right', fontsize=7, color='#d35400')
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_decomposition_bars_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_decomposition_bars_alpha{alpha}.png'}")
    
    # Plot 2: Arrow plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_alpha) * 0.4)))
    
    for i, (cc, ic, dc) in enumerate(zip(correct_changes, incorrect_changes, df_alpha['mean_diff_change'].values)):
        if abs(cc) > 0.001:
            ax.annotate('', xy=(cc, i + 0.15), xytext=(0, i + 0.15),
                        arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
        if abs(ic) > 0.001:
            ax.annotate('', xy=(ic, i - 0.15), xytext=(0, i - 0.15),
                        arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
        
        net_color = '#2ecc71' if dc > 0 else '#e74c3c'
        ax.plot(dc, i, 'o', color=net_color, markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Change in Log-Probability', fontsize=11)
    ax.set_title(f'Effect Arrows: Blue=correct, Orange=incorrect, Dot=net (α={alpha})', fontsize=12)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498db', lw=2, label='ΔLogP(correct)'),
        Line2D([0], [0], color='#e67e22', lw=2, label='ΔLogP(incorrect)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Net effect (+)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Net effect (-)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_decomposition_arrows_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_decomposition_arrows_alpha{alpha}.png'}")


def plot_mechanism_summary(
    df: pd.DataFrame,
    output_path: Path,
    alpha: float,
):
    """Create plots categorizing directions by their mechanism."""
    df_alpha = df[df["alpha"] == alpha].copy()
    
    def categorize_mechanism(row):
        cc = row['mean_correct_change']
        ic = row['mean_incorrect_change']
        dc = row['mean_diff_change']
        threshold = 0.02
        
        if dc > 0:
            if cc > threshold and abs(ic) < threshold:
                return 'Boost correct only'
            elif ic < -threshold and abs(cc) < threshold:
                return 'Suppress incorrect only'
            elif cc > threshold and ic < -threshold:
                return 'Best: boost correct & suppress incorrect'
            elif cc > 0 and ic > 0 and cc > ic:
                return 'Both increase, correct more'
            elif cc < 0 and ic < 0 and ic < cc:
                return 'Both decrease, incorrect more'
            else:
                return 'Other (positive)'
        else:
            if ic > threshold and abs(cc) < threshold:
                return 'Bad: boost incorrect only'
            elif cc < -threshold and abs(ic) < threshold:
                return 'Bad: suppress correct only'
            elif ic > 0 and cc > 0 and ic > cc:
                return 'Bad: both increase, incorrect more'
            else:
                return 'Bad: other'
    
    df_alpha['mechanism'] = df_alpha.apply(categorize_mechanism, axis=1)
    
    mechanism_colors = {
        'Best: boost correct & suppress incorrect': '#27ae60',
        'Suppress incorrect only': '#2ecc71',
        'Boost correct only': '#82e0aa',
        'Both increase, correct more': '#85c1e9',
        'Both decrease, incorrect more': '#5dade2',
        'Other (positive)': '#aab7b8',
        'Bad: boost incorrect only': '#e74c3c',
        'Bad: suppress correct only': '#c0392b',
        'Bad: both increase, incorrect more': '#f1948a',
        'Bad: other': '#d5a6bd',
    }
    
    # Plot 1: Scatter by mechanism
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for mechanism, group in df_alpha.groupby('mechanism'):
        color = mechanism_colors.get(mechanism, '#aab7b8')
        ax.scatter(group['mean_incorrect_change'], group['mean_correct_change'],
                   c=color, s=150, label=mechanism, edgecolors='black', linewidth=0.5, alpha=0.8)
        
        for _, row in group.iterrows():
            short_name = shorten_direction_name(row['direction_name'])[:12]
            ax.annotate(short_name, (row['mean_incorrect_change'], row['mean_correct_change']),
                        fontsize=7, alpha=0.8, ha='center', va='bottom')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    
    ax.set_xlabel('ΔLogP(incorrect)', fontsize=12)
    ax.set_ylabel('ΔLogP(correct)', fontsize=12)
    ax.set_title(f'Mechanism Analysis: What causes the effect? (α={alpha})\nAbove diagonal = positive diff_change', fontsize=12)
    ax.legend(loc='upper left', fontsize=7, ncol=1)
    
    plt.tight_layout()
    plt.savefig(output_path / f'mechanism_scatter_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'mechanism_scatter_alpha{alpha}.png'}")
    
    # Plot 2: Bar chart of mechanism counts
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mechanism_counts = df_alpha['mechanism'].value_counts()
    good_mechanisms = [m for m in mechanism_counts.index if not m.startswith('Bad')]
    bad_mechanisms = [m for m in mechanism_counts.index if m.startswith('Bad')]
    all_mechanisms = good_mechanisms + bad_mechanisms
    counts = [mechanism_counts.get(m, 0) for m in all_mechanisms]
    colors = [mechanism_colors.get(m, '#aab7b8') for m in all_mechanisms]
    
    bars = ax.barh(range(len(all_mechanisms)), counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(all_mechanisms)))
    ax.set_yticklabels(all_mechanisms, fontsize=9)
    ax.set_xlabel('Number of directions', fontsize=11)
    ax.set_title(f'Mechanism Distribution (α={alpha})', fontsize=12)
    
    for i, count in enumerate(counts):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / f'mechanism_distribution_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'mechanism_distribution_alpha{alpha}.png'}")
    
    print(f"\n  Mechanism Summary (α={alpha}):")
    for mechanism in all_mechanisms:
        count = mechanism_counts.get(mechanism, 0)
        dirs_with_mech = df_alpha[df_alpha['mechanism'] == mechanism]['direction_name'].tolist()
        print(f"    {mechanism}: {count}")
        for d in dirs_with_mech:
            print(f"      - {shorten_direction_name(d)}")


def plot_waterfall_decomposition(
    df: pd.DataFrame,
    output_path: Path,
    alpha: float,
    top_n: int = 6,
):
    """Create waterfall-style plots showing the decomposition for top directions."""
    df_alpha = df[df["alpha"] == alpha].copy()
    df_sorted = df_alpha.sort_values('mean_diff_change', ascending=False)
    top_dirs = df_sorted.head(top_n)
    
    for _, row in top_dirs.iterrows():
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cc = row['mean_correct_change']
        ic = row['mean_incorrect_change']
        baseline_diff = row['mean_baseline_diff']
        final_diff = row['mean_intervened_diff']
        
        categories = ['Baseline\ndiff', '+Δcorrect', '-Δincorrect', 'Final\ndiff']
        cumulative = [baseline_diff, baseline_diff + cc, baseline_diff + cc - ic, final_diff]
        colors = ['#3498db', '#2ecc71' if cc > 0 else '#e74c3c', 
                  '#2ecc71' if ic < 0 else '#e74c3c', '#9b59b6']
        
        for i in range(len(categories)):
            if i == 0:
                ax.bar(i, cumulative[i], color=colors[i], edgecolor='black', linewidth=0.5)
            elif i == len(categories) - 1:
                ax.bar(i, cumulative[i], color=colors[i], edgecolor='black', linewidth=0.5)
            else:
                bottom = cumulative[i-1]
                height = cumulative[i] - cumulative[i-1]
                ax.bar(i, height, bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.5)
                ax.plot([i-0.4, i-0.1], [cumulative[i-1], cumulative[i-1]], 'k-', linewidth=1)
        
        ax.text(0, cumulative[0]/2, f'{cumulative[0]:.2f}', ha='center', va='center', fontsize=10)
        ax.text(1, cumulative[0] + (cumulative[1]-cumulative[0])/2, f'{cc:+.3f}', ha='center', va='center', fontsize=10)
        ax.text(2, cumulative[1] + (cumulative[2]-cumulative[1])/2, f'{-ic:+.3f}', ha='center', va='center', fontsize=10)
        ax.text(3, cumulative[3]/2, f'{cumulative[3]:.2f}', ha='center', va='center', fontsize=10)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        short_name = shorten_direction_name(row['direction_name'])
        diff_change = row['mean_diff_change']
        ax.set_title(f'{short_name}: Waterfall Decomposition (α={alpha})\nΔdiff = {diff_change:+.3f}', fontsize=12,
                    color='green' if diff_change > 0 else 'red')
        ax.set_ylabel('LogP(correct) - LogP(incorrect)', fontsize=11)
        
        plt.tight_layout()
        clean_name = row['direction_name'].replace('[', '_').replace(']', '').replace('/', '_')
        plt.savefig(output_path / f'waterfall_{clean_name}_alpha{alpha}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(top_dirs)} waterfall plots")


def create_combined_direction_plot(
    df: pd.DataFrame,
    all_results: Dict[str, List[QAInterventionResult]],
    output_path: Path,
    alpha: float,
):
    """Create a grid of mini scatter plots showing all directions side by side."""
    df_alpha = df[df["alpha"] == alpha].copy()
    df_alpha = df_alpha.sort_values('mean_diff_change', ascending=False)
    
    n_directions = len(df_alpha)
    n_cols = 4
    n_rows = (n_directions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_directions > 1 else [axes]
    
    for idx, (_, row) in enumerate(df_alpha.iterrows()):
        ax = axes[idx]
        dir_name = row['direction_name']
        key = f"{dir_name}_alpha{alpha}"
        
        if key not in all_results:
            ax.set_visible(False)
            continue
        
        results = all_results[key]
        baseline_diffs = [r.baseline_diff for r in results]
        intervened_diffs = [r.intervened_diff for r in results]
        diff_changes = [r.diff_change for r in results]
        was_intervened = [r.was_intervened for r in results]
        
        colors = []
        for dc, wi in zip(diff_changes, was_intervened):
            if not wi:
                colors.append('#aaaaaa')
            elif dc > 0:
                colors.append('#2ecc71')
            else:
                colors.append('#e74c3c')
        
        ax.scatter(baseline_diffs, intervened_diffs, c=colors, alpha=0.5, s=20, edgecolors='none')
        
        min_val = min(min(baseline_diffs), min(intervened_diffs))
        max_val = max(max(baseline_diffs), max(intervened_diffs))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
        
        n_intervened = sum(was_intervened)
        n_improved = sum(1 for dc, wi in zip(diff_changes, was_intervened) if dc > 0 and wi)
        mean_change = row['mean_diff_change']
        short_name = shorten_direction_name(dir_name)[:20]
        
        color = 'green' if mean_change > 0 else 'red'
        ax.set_title(f'{short_name}\nΔ={mean_change:+.3f} ({n_improved}/{n_intervened} int)', fontsize=9, color=color)
        ax.set_xlabel('Baseline', fontsize=7)
        ax.set_ylabel('Intervened', fontsize=7)
        ax.tick_params(labelsize=6)
    
    for idx in range(n_directions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Per-Example Effects for All Directions (α={alpha})\nGreen = improved, Red = hurt, Gray = not intervened', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / f'all_directions_grid_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'all_directions_grid_alpha{alpha}.png'}")


def create_histogram_grid(
    df: pd.DataFrame,
    all_results: Dict[str, List[QAInterventionResult]],
    output_path: Path,
    alpha: float,
):
    """Create a grid of histograms showing diff_change distribution for all directions."""
    df_alpha = df[df["alpha"] == alpha].copy()
    df_alpha = df_alpha.sort_values('mean_diff_change', ascending=False)
    
    n_directions = len(df_alpha)
    n_cols = 4
    n_rows = (n_directions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_directions > 1 else [axes]
    
    all_changes = []
    for _, row in df_alpha.iterrows():
        key = f"{row['direction_name']}_alpha{alpha}"
        if key in all_results:
            all_changes.extend([r.diff_change for r in all_results[key] if r.was_intervened])
    
    if all_changes:
        x_min, x_max = np.percentile(all_changes, [1, 99])
        x_range = max(abs(x_min), abs(x_max)) * 1.2
    else:
        x_range = 1.0
    
    for idx, (_, row) in enumerate(df_alpha.iterrows()):
        ax = axes[idx]
        dir_name = row['direction_name']
        key = f"{dir_name}_alpha{alpha}"
        
        if key not in all_results:
            ax.set_visible(False)
            continue
        
        results = all_results[key]
        diff_changes = [r.diff_change for r in results if r.was_intervened]
        
        if diff_changes:
            ax.hist(diff_changes, bins=20, edgecolor='black', alpha=0.7, 
                    color='#2ecc71' if row['mean_diff_change'] > 0 else '#e74c3c')
            ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
            ax.axvline(x=row['mean_diff_change'], color='blue', linewidth=2, linestyle='-')
        
        short_name = shorten_direction_name(dir_name)[:20]
        n_int = row.get('n_intervened', len(diff_changes))
        ax.set_title(f'{short_name}\nmean={row["mean_diff_change"]:+.3f} (n={n_int})', fontsize=9)
        ax.set_xlim(-x_range, x_range)
        ax.tick_params(labelsize=6)
    
    for idx in range(n_directions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Distribution of diff_change for All Directions (α={alpha})\nBlue line = mean, Black dashed = zero', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / f'all_directions_histograms_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'all_directions_histograms_alpha{alpha}.png'}")


# ============================================================================
# Baseline Ratio Analysis Plots
# ============================================================================

def plot_effect_by_baseline_ratio(
    df: pd.DataFrame,
    all_results: Dict[str, List[QAInterventionResult]],
    output_path: Path,
    alpha: float,
    n_bins: int = 10,
    top_directions: int = 5,
):
    """
    Plot mean diff_change as a function of baseline likelihood ratio.
    Generates separate plots for line chart and heatmap.
    """
    df_alpha = df[df["alpha"] == alpha].copy()
    df_sorted = df_alpha.sort_values('mean_diff_change', ascending=False)
    
    top_dirs = df_sorted.head(top_directions)['direction_name'].tolist()
    if 'general[0]' not in top_dirs:
        top_dirs.append('general[0]')
    
    all_baseline_diffs = []
    for dir_name in df_sorted['direction_name'].tolist():
        key = f"{dir_name}_alpha{alpha}"
        if key in all_results:
            all_baseline_diffs.extend([r.baseline_diff for r in all_results[key]])
    
    if not all_baseline_diffs:
        print("  No data for effect_by_baseline_ratio plot")
        return
    
    bin_edges, percentiles, bin_labels = compute_percentile_bins(all_baseline_diffs, n_bins)
    actual_n_bins = len(bin_edges) - 1
    
    ratio_1_bin = np.searchsorted(bin_edges[:-1], 0) - 0.5
    
    # Plot 1: Line plot for top directions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_dirs)))
    
    for dir_idx, dir_name in enumerate(top_dirs):
        key = f"{dir_name}_alpha{alpha}"
        if key not in all_results:
            continue
        
        results = all_results[key]
        binned = bin_results_by_baseline(results, bin_edges)
        
        bin_means = []
        bin_stds = []
        
        for bin_res in binned:
            if bin_res:
                changes = [r.diff_change for r in bin_res]
                bin_means.append(np.mean(changes))
                bin_stds.append(np.std(changes) / np.sqrt(len(changes)))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        x_pos = np.arange(actual_n_bins)
        short_name = shorten_direction_name(dir_name)
        is_general = 'general' in dir_name
        
        line_style = '--' if is_general else '-'
        marker = 'o' if is_general else 's'
        
        ax.errorbar(x_pos, bin_means, yerr=bin_stds, 
                    label=short_name, color=colors[dir_idx],
                    linestyle=line_style, marker=marker, markersize=6,
                    capsize=3, capthick=1, linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label='Ratio=1 (equal)')
    
    ax.set_xticks(np.arange(actual_n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('Mean diff_change ± SEM', fontsize=11)
    ax.set_title(f'Intervention Effect by Baseline Model Confidence (α={alpha})\n'
                 f'Left = model less confident, Right = model more confident', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_by_baseline_lines_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_by_baseline_lines_alpha{alpha}.png'}")
    
    # Plot 2: Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    all_dir_names = df_sorted['direction_name'].tolist()
    
    heatmap_data = []
    dir_labels = []
    
    for dir_name in all_dir_names:
        key = f"{dir_name}_alpha{alpha}"
        if key not in all_results:
            continue
        
        results = all_results[key]
        binned = bin_results_by_baseline(results, bin_edges)
        
        bin_means = []
        for bin_res in binned:
            if bin_res:
                bin_means.append(np.mean([r.diff_change for r in bin_res]))
            else:
                bin_means.append(np.nan)
        
        heatmap_data.append(bin_means)
        dir_labels.append(shorten_direction_name(dir_name))
    
    heatmap_array = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_array, aspect='auto', cmap='RdYlGn', 
                   vmin=-np.nanmax(np.abs(heatmap_array)), 
                   vmax=np.nanmax(np.abs(heatmap_array)))
    
    ax.set_xticks(np.arange(actual_n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(dir_labels)))
    ax.set_yticklabels(dir_labels, fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('Direction', fontsize=11)
    ax.set_title(f'Heatmap: Effect by Direction and Baseline (α={alpha})\n'
                 f'Green = helps, Red = hurts', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean diff_change', fontsize=10)
    
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='black', linestyle=':', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_by_baseline_heatmap_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_by_baseline_heatmap_alpha{alpha}.png'}")


def plot_effect_by_baseline_ratio_aggregate(
    df: pd.DataFrame,
    all_results: Dict[str, List[QAInterventionResult]],
    output_path: Path,
    alpha: float,
    n_bins: int = 10,
):
    """
    Plot aggregate effect across ALL directions as a function of baseline ratio.
    Generates three separate plots.
    """
    df_alpha = df[df["alpha"] == alpha].copy()
    
    general_results = []
    specific_results = []
    
    for _, row in df_alpha.iterrows():
        key = f"{row['direction_name']}_alpha{alpha}"
        if key not in all_results:
            continue
        
        results = all_results[key]
        if row['direction_type'] == 'general':
            general_results.extend(results)
        else:
            specific_results.extend(results)
    
    all_results_list = general_results + specific_results
    all_baseline_diffs = [r.baseline_diff for r in all_results_list]
    
    if not all_baseline_diffs:
        print("  No data for aggregate effect_by_baseline_ratio plot")
        return
    
    bin_edges, percentiles, bin_labels = compute_percentile_bins(all_baseline_diffs, n_bins)
    actual_n_bins = len(bin_edges) - 1
    
    ratio_1_bin = np.searchsorted(bin_edges[:-1], 0) - 0.5
    
    # Plot 1: General vs Specific
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for results_list, label, color, marker in [
        (general_results, 'General', '#9b59b6', 'o'),
        (specific_results, 'Specific', '#2ecc71', 's'),
    ]:
        if not results_list:
            continue
        
        binned = bin_results_by_baseline(results_list, bin_edges)
        
        bin_means = []
        bin_stds = []
        
        for bin_res in binned:
            if bin_res:
                changes = [r.diff_change for r in bin_res]
                bin_means.append(np.mean(changes))
                bin_stds.append(np.std(changes) / np.sqrt(len(changes)))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        x_pos = np.arange(actual_n_bins)
        n_intervened = len([r for r in results_list if r.was_intervened])
        ax.errorbar(x_pos, bin_means, yerr=bin_stds,
                    label=f'{label} (n={n_intervened})',
                    color=color, marker=marker, markersize=8, capsize=3, linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ratio=1')
    
    ax.set_xticks(np.arange(actual_n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('Mean diff_change ± SEM', fontsize=11)
    ax.set_title(f'General vs Specific by Baseline Confidence (α={alpha})', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_by_baseline_general_vs_specific_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_by_baseline_general_vs_specific_alpha{alpha}.png'}")
    
    # Plot 2: Correct vs Incorrect changes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_intervened = [r for r in all_results_list if r.was_intervened]
    binned = bin_results_by_baseline(all_intervened, bin_edges, intervened_only=False)
    
    correct_means, correct_sems = [], []
    incorrect_means, incorrect_sems = [], []
    
    for bin_res in binned:
        if bin_res:
            correct_changes = [r.intervened_correct_logprob - r.baseline_correct_logprob for r in bin_res]
            incorrect_changes = [r.intervened_incorrect_logprob - r.baseline_incorrect_logprob for r in bin_res]
            correct_means.append(np.mean(correct_changes))
            correct_sems.append(np.std(correct_changes) / np.sqrt(len(correct_changes)))
            incorrect_means.append(np.mean(incorrect_changes))
            incorrect_sems.append(np.std(incorrect_changes) / np.sqrt(len(incorrect_changes)))
        else:
            correct_means.append(np.nan)
            correct_sems.append(np.nan)
            incorrect_means.append(np.nan)
            incorrect_sems.append(np.nan)
    
    x_pos = np.arange(actual_n_bins)
    ax.errorbar(x_pos, correct_means, yerr=correct_sems, color='#3498db', 
                marker='o', markersize=8, capsize=3, linewidth=2, label='ΔLogP(correct)')
    ax.errorbar(x_pos, incorrect_means, yerr=incorrect_sems, color='#e67e22',
                marker='s', markersize=8, capsize=3, linewidth=2, label='ΔLogP(incorrect)')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ratio=1')
    
    ax.set_xticks(np.arange(actual_n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('Mean change in LogP ± SEM', fontsize=11)
    ax.set_title(f'Correct vs Incorrect Changes by Baseline (α={alpha})', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_by_baseline_correct_vs_incorrect_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_by_baseline_correct_vs_incorrect_alpha{alpha}.png'}")
    
    # Plot 3: Success rate
    fig, ax = plt.subplots(figsize=(12, 8))
    
    general_binned = bin_results_by_baseline(general_results, bin_edges)
    specific_binned = bin_results_by_baseline(specific_results, bin_edges)
    
    success_rates_general = []
    success_rates_specific = []
    
    for bin_res in general_binned:
        if bin_res:
            n_improved = sum(1 for r in bin_res if r.diff_change > 0)
            success_rates_general.append(100 * n_improved / len(bin_res))
        else:
            success_rates_general.append(np.nan)
    
    for bin_res in specific_binned:
        if bin_res:
            n_improved = sum(1 for r in bin_res if r.diff_change > 0)
            success_rates_specific.append(100 * n_improved / len(bin_res))
        else:
            success_rates_specific.append(np.nan)
    
    x_pos = np.arange(actual_n_bins)
    ax.plot(x_pos, success_rates_general, 'o-', color='#9b59b6',
            label='General', markersize=8, linewidth=2)
    ax.plot(x_pos, success_rates_specific, 's-', color='#2ecc71',
            label='Specific', markersize=8, linewidth=2)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (chance)')
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ratio=1')
    
    ax.set_xticks(np.arange(actual_n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('% of examples improved', fontsize=11)
    ax.set_title(f'Success Rate by Baseline Confidence (α={alpha})', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path / f'effect_by_baseline_success_rate_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / f'effect_by_baseline_success_rate_alpha{alpha}.png'}")


def plot_effect_by_baseline_ratio_per_direction(
    results: List[QAInterventionResult],
    direction_name: str,
    output_path: Path,
    alpha: float,
    n_bins: int = 10,
):
    """
    Plot effect by baseline ratio for a SINGLE direction.
    Generates three separate plots.
    """
    intervened_results = [r for r in results if r.was_intervened]
    
    if len(intervened_results) < 5:
        print(f"Skipping (only {len(intervened_results)} intervened)")
        return
    
    baseline_diffs = [r.baseline_diff for r in intervened_results]
    
    bin_edges, percentiles, bin_labels = compute_percentile_bins(baseline_diffs, n_bins)
    actual_n_bins = len(bin_edges) - 1
    
    if actual_n_bins < 2:
        print(f"Skipping (not enough bins)")
        return
    
    binned = bin_results_by_baseline(intervened_results, bin_edges, intervened_only=False)
    
    diff_means, diff_sems = [], []
    correct_means, correct_sems = [], []
    incorrect_means, incorrect_sems = [], []
    bin_counts = []
    success_rates = []
    
    for bin_res in binned:
        if bin_res:
            changes = [r.diff_change for r in bin_res]
            diff_means.append(np.mean(changes))
            diff_sems.append(np.std(changes) / np.sqrt(len(changes)))
            
            corr_changes = [r.intervened_correct_logprob - r.baseline_correct_logprob for r in bin_res]
            incorr_changes = [r.intervened_incorrect_logprob - r.baseline_incorrect_logprob for r in bin_res]
            correct_means.append(np.mean(corr_changes))
            correct_sems.append(np.std(corr_changes) / np.sqrt(len(corr_changes)))
            incorrect_means.append(np.mean(incorr_changes))
            incorrect_sems.append(np.std(incorr_changes) / np.sqrt(len(incorr_changes)))
            
            bin_counts.append(len(bin_res))
            n_improved = sum(1 for r in bin_res if r.diff_change > 0)
            success_rates.append(100 * n_improved / len(bin_res))
        else:
            diff_means.append(np.nan)
            diff_sems.append(np.nan)
            correct_means.append(np.nan)
            correct_sems.append(np.nan)
            incorrect_means.append(np.nan)
            incorrect_sems.append(np.nan)
            bin_counts.append(0)
            success_rates.append(np.nan)
    
    short_name = shorten_direction_name(direction_name)
    clean_name = direction_name.replace('[', '_').replace(']', '').replace('/', '_')
    overall_mean = np.mean([r.diff_change for r in intervened_results])
    overall_color = '#2ecc71' if overall_mean > 0 else '#e74c3c'
    
    ratio_1_bin = np.searchsorted(bin_edges[:-1], 0) - 0.5
    
    n_total = len(intervened_results)
    n_improved_total = sum(1 for r in intervened_results if r.diff_change > 0)
    title_suffix = f'\nOverall: mean={overall_mean:+.3f}, improved={n_improved_total}/{n_total} ({100*n_improved_total/n_total:.1f}%)'
    
    x_pos = np.arange(actual_n_bins)
    
    # Plot 1: diff_change
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.errorbar(x_pos, diff_means, yerr=diff_sems,
                color=overall_color, marker='o', markersize=10, 
                capsize=4, linewidth=2.5, markeredgecolor='black', markeredgewidth=1)
    
    for x, y, n in zip(x_pos, diff_means, bin_counts):
        if not np.isnan(y):
            ax.annotate(f'n={n}', (x, y), textcoords="offset points", 
                        xytext=(0, 12), ha='center', fontsize=9, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ratio=1')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('Mean diff_change ± SEM', fontsize=11)
    ax.set_title(f'{short_name}: Effect Size by Baseline Confidence (α={alpha}){title_suffix}', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'baseline_{clean_name}_diff_change_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Decomposition
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.errorbar(x_pos, correct_means, yerr=correct_sems,
                color='#3498db', marker='o', markersize=8, capsize=3, 
                linewidth=2, label='ΔLogP(correct)')
    ax.errorbar(x_pos, incorrect_means, yerr=incorrect_sems,
                color='#e67e22', marker='s', markersize=8, capsize=3,
                linewidth=2, label='ΔLogP(incorrect)')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ratio=1')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('Mean change in LogP ± SEM', fontsize=11)
    ax.set_title(f'{short_name}: Correct vs Incorrect Changes (α={alpha}){title_suffix}', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'baseline_{clean_name}_decomposition_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Success rate
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_colors = ['#2ecc71' if s > 50 else '#e74c3c' if not np.isnan(s) else '#cccccc' for s in success_rates]
    bars = ax.bar(x_pos, success_rates, color=bar_colors, 
                  edgecolor='black', linewidth=0.5, alpha=0.8)
    
    for i, (bar, n, s) in enumerate(zip(bars, bin_counts, success_rates)):
        if not np.isnan(s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{s:.0f}%\n(n={n})', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (chance)')
    if 0 <= ratio_1_bin < actual_n_bins:
        ax.axvline(x=ratio_1_bin, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ratio=1')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Baseline P(correct)/P(incorrect) percentile', fontsize=11)
    ax.set_ylabel('% of examples improved', fontsize=11)
    ax.set_title(f'{short_name}: Success Rate by Baseline (α={alpha}){title_suffix}', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 115)
    
    plt.tight_layout()
    plt.savefig(output_path / f'baseline_{clean_name}_success_rate_alpha{alpha}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 3 plots")


def plot_all_directions_baseline_ratio(
    df: pd.DataFrame,
    all_results: Dict[str, List[QAInterventionResult]],
    output_path: Path,
    alpha: float,
    n_bins: int = 10,
):
    """Generate baseline ratio plots for ALL directions."""
    baseline_plots_path = output_path / "baseline_ratio_plots"
    baseline_plots_path.mkdir(exist_ok=True, parents=True)
    
    df_alpha = df[df["alpha"] == alpha].copy()
    df_sorted = df_alpha.sort_values('mean_diff_change', ascending=False)
    
    print(f"\n  Generating baseline ratio plots for {len(df_sorted)} directions...")
    
    for _, row in df_sorted.iterrows():
        dir_name = row['direction_name']
        key = f"{dir_name}_alpha{alpha}"
        
        if key not in all_results:
            continue
        
        results = all_results[key]
        print(f"    {shorten_direction_name(dir_name)}: ", end="")
        plot_effect_by_baseline_ratio_per_direction(
            results, dir_name, baseline_plots_path, alpha, n_bins
        )
    
    print(f"\n  All baseline ratio plots saved to: {baseline_plots_path}")


def print_example_analysis(
    results: List[QAInterventionResult],
    direction_name: str,
    top_k: int = 5,
):
    """Print the examples that benefited most and least from the intervention."""
    intervened_results = [r for r in results if r.was_intervened]
    
    if not intervened_results:
        print(f"\n{'='*70}")
        print(f"EXAMPLE ANALYSIS: {direction_name}")
        print(f"{'='*70}")
        print("  No examples were intervened (all above threshold).")
        return
    
    sorted_results = sorted(intervened_results, key=lambda r: r.diff_change, reverse=True)
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE ANALYSIS: {direction_name}")
    print(f"Intervened: {len(intervened_results)}/{len(results)} examples")
    print(f"{'='*70}")
    
    print(f"\n📈 TOP {top_k} MOST IMPROVED (diff_change > 0 = better discrimination):")
    print("-" * 70)
    for i, r in enumerate(sorted_results[:top_k]):
        print(f"\n{i+1}. diff_change = {r.diff_change:+.4f}")
        print(f"   Q: {r.question[:80]}{'...' if len(r.question) > 80 else ''}")
        print(f"   ✓ Correct: {r.correct_answer[:50]}{'...' if len(r.correct_answer) > 50 else ''}")
        print(f"   ✗ Incorrect: {r.incorrect_answer[:50]}{'...' if len(r.incorrect_answer) > 50 else ''}")
        print(f"   Baseline: LogP(correct)={r.baseline_correct_logprob:.3f}, LogP(incorrect)={r.baseline_incorrect_logprob:.3f}, diff={r.baseline_diff:.3f}")
        print(f"   After:    LogP(correct)={r.intervened_correct_logprob:.3f}, LogP(incorrect)={r.intervened_incorrect_logprob:.3f}, diff={r.intervened_diff:.3f}")
        correct_change = r.intervened_correct_logprob - r.baseline_correct_logprob
        incorrect_change = r.intervened_incorrect_logprob - r.baseline_incorrect_logprob
        print(f"   Changes:  correct {correct_change:+.3f}, incorrect {incorrect_change:+.3f}")
    
    print(f"\n📉 TOP {top_k} MOST HURT (diff_change < 0 = worse discrimination):")
    print("-" * 70)
    for i, r in enumerate(sorted_results[-top_k:][::-1]):
        print(f"\n{i+1}. diff_change = {r.diff_change:+.4f}")
        print(f"   Q: {r.question[:80]}{'...' if len(r.question) > 80 else ''}")
        print(f"   ✓ Correct: {r.correct_answer[:50]}{'...' if len(r.correct_answer) > 50 else ''}")
        print(f"   ✗ Incorrect: {r.incorrect_answer[:50]}{'...' if len(r.incorrect_answer) > 50 else ''}")
        print(f"   Baseline: LogP(correct)={r.baseline_correct_logprob:.3f}, LogP(incorrect)={r.baseline_incorrect_logprob:.3f}, diff={r.baseline_diff:.3f}")
        print(f"   After:    LogP(correct)={r.intervened_correct_logprob:.3f}, LogP(incorrect)={r.intervened_incorrect_logprob:.3f}, diff={r.intervened_diff:.3f}")
        correct_change = r.intervened_correct_logprob - r.baseline_correct_logprob
        incorrect_change = r.intervened_incorrect_logprob - r.baseline_incorrect_logprob
        print(f"   Changes:  correct {correct_change:+.3f}, incorrect {incorrect_change:+.3f}")
    
    diff_changes = [r.diff_change for r in intervened_results]
    n_improved = sum(1 for dc in diff_changes if dc > 0)
    n_hurt = sum(1 for dc in diff_changes if dc < 0)
    
    print(f"\n📊 SUMMARY (intervened examples only):")
    print(f"   Total intervened: {len(intervened_results)}/{len(results)}")
    print(f"   Improved (diff_change > 0): {n_improved} ({100*n_improved/len(intervened_results):.1f}%)")
    print(f"   Hurt (diff_change < 0): {n_hurt} ({100*n_hurt/len(intervened_results):.1f}%)")
    print(f"   Mean diff_change: {np.mean(diff_changes):+.4f}")
    print(f"   Median diff_change: {np.median(diff_changes):+.4f}")
    print(f"   Std diff_change: {np.std(diff_changes):.4f}")


# ============================================================================
# Main Assessment Class
# ============================================================================

class CausalDirectionAssessment:
    """Assess causal importance of truth directions via bias modification."""
    
    def __init__(
        self,
        model_name: str,
        directions_path: str,
        layer_idx: int,
        model_type: str = "llama",
        bias_location: str = "mlp_out",
        device: str = "cuda",
        use_original_space: bool = True,
        conditional: bool = False,
        conditional_threshold: float = 1.0,
    ):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.model_type = model_type
        self.bias_location = bias_location
        self.device = device
        self.conditional = conditional
        self.conditional_threshold = conditional_threshold
        self.conditional_log_threshold = np.log(conditional_threshold) if conditional_threshold > 0 else float('-inf')
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        
        bias_param, path = get_layer_bias_param(self.model, layer_idx, model_type, bias_location)
        if bias_param is None:
            print(f"Note: No existing bias at {path}")
            print(f"      Bias will be created during intervention and removed after")
        else:
            print(f"Found existing bias at {path}, shape: {bias_param.shape}")
        
        if self.conditional:
            print(f"\nConditional intervention ENABLED:")
            print(f"  Only intervene when P(correct)/P(incorrect) < {conditional_threshold}")
            print(f"  Equivalent to: LogP(correct) - LogP(incorrect) < {self.conditional_log_threshold:.4f}")
        
        self.directions = self._load_directions(directions_path, use_original_space)
        self.assessments: Dict[str, QADirectionAssessment] = {}
        self.all_results: Dict[str, List[QAInterventionResult]] = {}
    
    def _load_directions(self, path: str, use_original_space: bool) -> Dict[str, np.ndarray]:
        """Load directions from file."""
        path = Path(path)
        if path.is_dir():
            if use_original_space and (path / "directions_original_space.npz").exists():
                path = path / "directions_original_space.npz"
            else:
                path = path / "directions.npz"
        
        print(f"Loading directions from: {path}")
        data = dict(np.load(path, allow_pickle=True))
        
        directions = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 2:
                print(f"  {key}: {value.shape}")
                directions[key] = value
        return directions
    
    def _should_intervene(self, baseline_diff: float) -> bool:
        """Determine if we should intervene based on conditional settings."""
        if not self.conditional:
            return True
        return baseline_diff < self.conditional_log_threshold
    
    def assess_direction_qa(
        self,
        direction_name: str,
        direction_idx: int,
        examples: List[Dict[str, str]],
        alphas: List[float] = [0.5, 1.0, 2.0, 5.0],
        prompt_template: Optional[str] = None,
    ) -> Dict[float, QADirectionAssessment]:
        """Assess causal effect of a direction using QA pairs with correct/incorrect answers."""
        if direction_name not in self.directions:
            raise ValueError(f"Direction '{direction_name}' not found.")
        
        direction_set = self.directions[direction_name]
        if direction_idx >= direction_set.shape[0]:
            raise ValueError(f"Direction index {direction_idx} out of range.")
        
        direction = direction_set[direction_idx]
        direction_type = "general" if direction_name == "general" else "specific"
        source_domain = None if direction_name == "general" else direction_name.replace("specific_", "")
        
        print(f"\nComputing baselines for {direction_name}[{direction_idx}]...")
        baselines = []
        for ex in tqdm(examples, desc="Baseline"):
            prompt = format_qa_prompt(ex["question"], prompt_template)
            correct_answer = " " + ex["correct_answer"]
            incorrect_answer = " " + ex["incorrect_answer"]
            
            correct_logprob = compute_answer_logprob(self.model, self.tokenizer, prompt, correct_answer, self.device)
            incorrect_logprob = compute_answer_logprob(self.model, self.tokenizer, prompt, incorrect_answer, self.device)
            
            baselines.append({
                "question": ex["question"], "correct_answer": ex["correct_answer"],
                "incorrect_answer": ex["incorrect_answer"],
                "answer_type": ex.get("answer_type", "unknown"), "prompt": prompt,
                "correct_logprob": correct_logprob, "incorrect_logprob": incorrect_logprob,
                "diff": correct_logprob - incorrect_logprob,
            })
        
        results_by_alpha = {}
        for alpha in alphas:
            print(f"\nAssessing {direction_name}[{direction_idx}] with alpha={alpha}")
            results = []
            n_intervened = 0
            n_skipped = 0
            
            for baseline_data in tqdm(baselines, desc=f"alpha={alpha}"):
                should_intervene = self._should_intervene(baseline_data["diff"])
                
                correct_answer = " " + baseline_data["correct_answer"]
                incorrect_answer = " " + baseline_data["incorrect_answer"]
                
                if should_intervene:
                    with bias_intervention(self.model, self.layer_idx, direction, alpha=alpha,
                                           model_type=self.model_type, bias_location=self.bias_location,
                                           verbose=(n_intervened == 0 and alpha == alphas[0])):
                        correct_logprob = compute_answer_logprob(self.model, self.tokenizer, 
                                                                  baseline_data["prompt"], correct_answer, self.device)
                        incorrect_logprob = compute_answer_logprob(self.model, self.tokenizer, 
                                                                    baseline_data["prompt"], incorrect_answer, self.device)
                    n_intervened += 1
                else:
                    correct_logprob = baseline_data["correct_logprob"]
                    incorrect_logprob = baseline_data["incorrect_logprob"]
                    n_skipped += 1
                
                intervened_diff = correct_logprob - incorrect_logprob
                
                results.append(QAInterventionResult(
                    question=baseline_data["question"],
                    correct_answer=baseline_data["correct_answer"],
                    incorrect_answer=baseline_data["incorrect_answer"],
                    answer_type=baseline_data["answer_type"],
                    baseline_correct_logprob=baseline_data["correct_logprob"],
                    baseline_incorrect_logprob=baseline_data["incorrect_logprob"],
                    baseline_diff=baseline_data["diff"],
                    intervened_correct_logprob=correct_logprob,
                    intervened_incorrect_logprob=incorrect_logprob,
                    intervened_diff=intervened_diff,
                    diff_change=intervened_diff - baseline_data["diff"],
                    direction_name=f"{direction_name}[{direction_idx}]", alpha=alpha,
                    was_intervened=should_intervene,
                ))
            
            key = f"{direction_name}[{direction_idx}]_alpha{alpha}"
            self.all_results[key] = results
            
            intervened_results = [r for r in results if r.was_intervened]
            if intervened_results:
                diff_changes = [r.diff_change for r in intervened_results]
                baseline_diffs = [r.baseline_diff for r in intervened_results]
                intervened_diffs = [r.intervened_diff for r in intervened_results]
                correct_changes = [r.intervened_correct_logprob - r.baseline_correct_logprob for r in intervened_results]
                incorrect_changes = [r.intervened_incorrect_logprob - r.baseline_incorrect_logprob for r in intervened_results]
            else:
                diff_changes = [0]
                baseline_diffs = [0]
                intervened_diffs = [0]
                correct_changes = [0]
                incorrect_changes = [0]
            
            from scipy.stats import wilcoxon
            try:
                _, p_value = wilcoxon(diff_changes)
            except:
                p_value = 1.0
            
            assessment = QADirectionAssessment(
                direction_name=f"{direction_name}[{direction_idx}]",
                direction_type=direction_type, source_domain=source_domain, alpha=alpha,
                n_examples=len(results), mean_baseline_diff=np.mean(baseline_diffs),
                std_baseline_diff=np.std(baseline_diffs), mean_intervened_diff=np.mean(intervened_diffs),
                mean_diff_change=np.mean(diff_changes), std_diff_change=np.std(diff_changes),
                median_diff_change=np.median(diff_changes), mean_correct_change=np.mean(correct_changes),
                mean_incorrect_change=np.mean(incorrect_changes), p_value=p_value,
                n_intervened=n_intervened, n_skipped=n_skipped, results=results,
            )
            results_by_alpha[alpha] = assessment
            
            print(f"  Intervened: {n_intervened}/{len(results)} (skipped {n_skipped})")
            print(f"  Mean baseline diff (correct-incorrect): {assessment.mean_baseline_diff:.4f}")
            print(f"  Mean intervened diff: {assessment.mean_intervened_diff:.4f}")
            print(f"  Mean diff change: {assessment.mean_diff_change:+.4f}")
            print(f"  Mean correct logprob change: {assessment.mean_correct_change:+.4f}")
            print(f"  Mean incorrect logprob change: {assessment.mean_incorrect_change:+.4f}")
            print(f"  p-value: {assessment.p_value:.4e}")
        
        return results_by_alpha
    
    def assess_all_directions_qa(
        self,
        examples: List[Dict[str, str]],
        alphas: List[float] = [1.0, 2.0],
        max_directions_per_set: int = 3,
        prompt_template: Optional[str] = None,
    ) -> pd.DataFrame:
        """Assess all directions using QA pairs with likelihood ratio metric."""
        all_results = []
        for direction_name, direction_set in self.directions.items():
            n_directions = min(max_directions_per_set, direction_set.shape[0])
            for direction_idx in range(n_directions):
                results_by_alpha = self.assess_direction_qa(
                    direction_name=direction_name, direction_idx=direction_idx,
                    examples=examples, alphas=alphas, prompt_template=prompt_template,
                )
                for alpha, assessment in results_by_alpha.items():
                    all_results.append({
                        "direction_name": assessment.direction_name,
                        "direction_type": assessment.direction_type,
                        "source_domain": assessment.source_domain, "alpha": alpha,
                        "n_examples": assessment.n_examples,
                        "n_intervened": assessment.n_intervened,
                        "n_skipped": assessment.n_skipped,
                        "mean_baseline_diff": assessment.mean_baseline_diff,
                        "mean_intervened_diff": assessment.mean_intervened_diff,
                        "mean_diff_change": assessment.mean_diff_change,
                        "std_diff_change": assessment.std_diff_change,
                        "median_diff_change": assessment.median_diff_change,
                        "mean_correct_change": assessment.mean_correct_change,
                        "mean_incorrect_change": assessment.mean_incorrect_change,
                        "p_value": assessment.p_value,
                    })
        return pd.DataFrame(all_results)
    
    def compare_general_vs_specific_qa(self, df: pd.DataFrame, alpha: float = 1.0) -> Dict:
        """Compare general vs specific directions on QA pairs."""
        df_alpha = df[df["alpha"] == alpha]
        general = df_alpha[df_alpha["direction_type"] == "general"]
        specific = df_alpha[df_alpha["direction_type"] == "specific"]
        
        comparison = {
            "alpha": alpha,
            "general": {
                "n_directions": len(general),
                "mean_diff_change": general["mean_diff_change"].mean() if len(general) > 0 else 0,
                "std_diff_change": general["mean_diff_change"].std() if len(general) > 0 else 0,
                "mean_correct_change": general["mean_correct_change"].mean() if len(general) > 0 else 0,
                "mean_incorrect_change": general["mean_incorrect_change"].mean() if len(general) > 0 else 0,
            },
            "specific": {
                "n_directions": len(specific),
                "mean_diff_change": specific["mean_diff_change"].mean() if len(specific) > 0 else 0,
                "std_diff_change": specific["mean_diff_change"].std() if len(specific) > 0 else 0,
                "mean_correct_change": specific["mean_correct_change"].mean() if len(specific) > 0 else 0,
                "mean_incorrect_change": specific["mean_incorrect_change"].mean() if len(specific) > 0 else 0,
            },
        }
        
        from scipy.stats import mannwhitneyu
        try:
            if len(general) > 0 and len(specific) > 0:
                _, p_value = mannwhitneyu(general["mean_diff_change"].values, 
                                           specific["mean_diff_change"].values, alternative="two-sided")
                comparison["comparison_p_value"] = p_value
            else:
                comparison["comparison_p_value"] = None
        except:
            comparison["comparison_p_value"] = None
        return comparison


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal Assessment via Bias Modification")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--directions", "-d", type=str, default="results")
    parser.add_argument("--layer", "-l", type=int, default=15)
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "mistral", "gpt2", "qwen2"])
    parser.add_argument("--bias-location", type=str, default="mlp_out", choices=["mlp_out", "attn_out"])
    parser.add_argument("--n-samples", "-n", type=int, default=1024)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-2.0])
    parser.add_argument("--max-directions", type=int, default=1)
    parser.add_argument("--output", "-o", type=str, default="causal_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--conditional", action="store_true",
                        help="Enable conditional intervention: only intervene when model favors incorrect answer")
    parser.add_argument("--conditional-threshold", type=float, default=1.0,
                        help="Threshold for conditional intervention. Intervene only when "
                             "P(correct)/P(incorrect) < threshold. Default 1.0 means intervene "
                             "only when model originally favors incorrect answer.")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of bins for baseline ratio analysis")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plots_path = output_path / "direction_plots"
    plots_path.mkdir(exist_ok=True, parents=True)
    
    assessor = CausalDirectionAssessment(
        model_name=args.model, directions_path=args.directions, layer_idx=args.layer,
        model_type=args.model_type, bias_location=args.bias_location,
        device=args.device, use_original_space=True,
        conditional=args.conditional, conditional_threshold=args.conditional_threshold,
    )
    
    print(f"\nLoading SimpleQA dataset with distractors (n={args.n_samples})...")
    examples = load_simpleqa_with_distractors(n_samples=args.n_samples, seed=args.seed)
    print(f"Loaded {len(examples)} examples with correct/incorrect answer pairs")
    
    type_counts = defaultdict(int)
    for ex in examples:
        type_counts[ex["answer_type"]] += 1
    print("\nAnswer type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    
    print("\n" + "=" * 60)
    print("Running causal assessment on SimpleQA (LIKELIHOOD RATIO)...")
    print(f"Intervening on layer {args.layer}, bias location: {args.bias_location}")
    print("Metric: Change in LogP(correct) - LogP(incorrect)")
    if args.conditional:
        print(f"CONDITIONAL MODE: Only intervene when P(correct)/P(incorrect) < {args.conditional_threshold}")
    print("=" * 60)
    
    df = assessor.assess_all_directions_qa(
        examples=examples, alphas=args.alphas, max_directions_per_set=args.max_directions,
    )
    
    df.to_csv(output_path / "causal_results_qa.csv", index=False)
    print(f"\nSaved results to {output_path / 'causal_results_qa.csv'}")
    
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    for alpha in args.alphas:
        print(f"\n--- Plots for alpha={alpha} ---")
        plot_direction_comparison(df, output_path, alpha, title_prefix="SimpleQA: ")
        plot_correct_vs_incorrect_changes(df, output_path, alpha)
        plot_effect_decomposition(df, output_path, alpha)
        plot_mechanism_summary(df, output_path, alpha)
        plot_waterfall_decomposition(df, output_path, alpha)
        
        # Baseline ratio analysis
        plot_effect_by_baseline_ratio(df, assessor.all_results, output_path, alpha, n_bins=args.n_bins)
        plot_effect_by_baseline_ratio_aggregate(df, assessor.all_results, output_path, alpha, n_bins=args.n_bins)
        plot_all_directions_baseline_ratio(df, assessor.all_results, output_path, alpha, n_bins=args.n_bins)
    
    print("\n" + "=" * 60)
    print("GENERATING PER-DIRECTION PLOTS")
    print("=" * 60)
    
    for alpha in args.alphas:
        df_alpha = df[df["alpha"] == alpha]
        df_sorted = df_alpha.sort_values('mean_diff_change', ascending=False)
        
        for _, row in df_sorted.iterrows():
            dir_name = row['direction_name']
            key = f"{dir_name}_alpha{alpha}"
            if key in assessor.all_results:
                results = assessor.all_results[key]
                print(f"  Plotting: {dir_name} (diff_change={row['mean_diff_change']:+.4f}, "
                      f"intervened={row['n_intervened']}/{row['n_examples']})")
                plot_per_example_effects(results, dir_name, plots_path, alpha)
    
    print("\n  Creating combined summary plots...")
    for alpha in args.alphas:
        create_combined_direction_plot(df, assessor.all_results, output_path, alpha)
        create_histogram_grid(df, assessor.all_results, output_path, alpha)
    
    print("\n" + "=" * 60)
    print("EXAMPLE ANALYSIS")
    print("=" * 60)
    
    for alpha in args.alphas:
        df_alpha = df[df["alpha"] == alpha]
        best_idx = df_alpha['mean_diff_change'].idxmax()
        worst_idx = df_alpha['mean_diff_change'].idxmin()
        best_dir = df_alpha.loc[best_idx, 'direction_name']
        worst_dir = df_alpha.loc[worst_idx, 'direction_name']
        
        key = f"{best_dir}_alpha{alpha}"
        if key in assessor.all_results:
            print_example_analysis(assessor.all_results[key], f"{best_dir} (BEST)", args.top_k)
        
        key = f"{worst_dir}_alpha{alpha}"
        if key in assessor.all_results:
            print_example_analysis(assessor.all_results[key], f"{worst_dir} (WORST)", args.top_k)
        
        key = f"general[0]_alpha{alpha}"
        if key in assessor.all_results:
            print_example_analysis(assessor.all_results[key], "general[0]", args.top_k)
    
    print("\n" + "=" * 60)
    print("GENERAL vs DOMAIN-SPECIFIC COMPARISON (SimpleQA)")
    print("=" * 60)
    
    for alpha in args.alphas:
        comparison = assessor.compare_general_vs_specific_qa(df, alpha=alpha)
        print(f"\n--- Alpha = {alpha} ---")
        print(f"General ({comparison['general']['n_directions']} dirs):")
        print(f"  Mean diff change (correct-incorrect): {comparison['general']['mean_diff_change']:+.4f}")
        print(f"  Mean correct logprob change: {comparison['general']['mean_correct_change']:+.4f}")
        print(f"  Mean incorrect logprob change: {comparison['general']['mean_incorrect_change']:+.4f}")
        print(f"Specific ({comparison['specific']['n_directions']} dirs):")
        print(f"  Mean diff change (correct-incorrect): {comparison['specific']['mean_diff_change']:+.4f}")
        print(f"  Mean correct logprob change: {comparison['specific']['mean_correct_change']:+.4f}")
        print(f"  Mean incorrect logprob change: {comparison['specific']['mean_incorrect_change']:+.4f}")
        if comparison['comparison_p_value'] is not None:
            print(f"Comparison p-value: {comparison['comparison_p_value']:.4e}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
diff_change = Δ(LogP(correct) - LogP(incorrect))

  diff_change > 0: Intervention INCREASED P(correct)/P(incorrect)
                   → Direction improves discrimination toward correct answer
                   
  diff_change < 0: Intervention DECREASED P(correct)/P(incorrect)
                   → Direction hurts discrimination

Also check:
  mean_correct_change: How much LogP(correct) changed
  mean_incorrect_change: How much LogP(incorrect) changed

If both increase but incorrect increases MORE → diff_change < 0 (bad)
If both increase but correct increases MORE → diff_change > 0 (good)
If correct increases, incorrect decreases → diff_change > 0 (very good!)
""")
    
    if args.conditional:
        print("\n" + "=" * 60)
        print("CONDITIONAL INTERVENTION SUMMARY")
        print("=" * 60)
        print(f"Threshold: P(correct)/P(incorrect) < {args.conditional_threshold}")
        print(f"Only examples where model originally favored incorrect answer were intervened.")
        total_intervened = df['n_intervened'].sum()
        total_examples = df['n_examples'].sum()
        print(f"Total intervened across all directions: {total_intervened}/{total_examples} "
              f"({100*total_intervened/total_examples:.1f}%)")
    
    print(f"\n📊 All plots saved to: {output_path}")
    print(f"📊 Individual direction plots saved to: {plots_path}")
    print(f"📊 Baseline ratio plots saved to: {output_path / 'baseline_ratio_plots'}")
    print("\nDone!")


if __name__ == "__main__":
    main()