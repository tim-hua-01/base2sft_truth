"""
LEACE concept erasure experiment script.

Trains LEACE erasers on specified tasks, evaluates probes before/after erasure,
and generates heatmap visualizations of cross-task generalization.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import h5py
import time
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from concept_erasure import LeaceEraser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config
from src.models import get_model_and_tokenizer
from src import utils

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEED = 42
MODEL_NAME = "llama-70b-3.3"
LAYER_NAME = "layer_33"
FEATURE_TYPE = "average"
RESULTS_CSV = f"./results/results_{MODEL_NAME}_all_layers.csv"
PROBE_DIR = f"./results/probes/{MODEL_NAME}"
FEATS_PATH = f"./results/extracted_feats_all_layers_{MODEL_NAME}.h5"
SAVE_PATH = "./results/inlp/leace_llama70b_layer33_FLEED_RP_IT_SB.npy"
PLOT_DIR = "./plots"

ERASE_TASKS = [
    "claims__definitional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "claims__logical_gemini_600_full",
    "ethics__commonsense",
]

TRAIN_TASKS = [
    "claims__definitional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__logical_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "ethics__commonsense",
]

TEST_TASKS = [
    "claims__definitional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__logical_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "ethics__commonsense",
    "roleplaying__plain",
    "insider_trading__upscale",
    "sandbagging_v2__wmdp_mmlu",
]

CUSTOM_LABELS = {
    "got__best": "Marks & Tegmark",
    "repe_honesty__plain": "Goldowsky-Dill et al.",
    "claims__definitional_gemini_600_full": "Definitional",
    "claims__evidential_gemini_600_full": "Empirical",
    "claims__fictional_gemini_600_full": "Fictional",
    "claims__logical_gemini_600_full": "Logical",
    "claims__definitional_gemini_full": "definitional",
    "claims__evidential_gemini_full": "evidential",
    "claims__fictional_gemini_full": "fictional",
    "claims__logical_gemini_full": "logical",
    "ethics__commonsense": "Ethical",
    "roleplaying__plain": "Roleplaying",
    "insider_trading__upscale": "Insider Trading",
    "sandbagging_v2__wmdp_mmlu": "Sandbagging",
}

# Task lists used for plotting (different order from training lists)
PLOT_INLP_TASKS = [
    "claims__definitional_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__logical_gemini_600_full",
    "ethics__commonsense",
]
PLOT_TRAIN_TASKS = [
    "claims__definitional_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__logical_gemini_600_full",
    "ethics__commonsense",
]
PLOT_TEST_TASKS = [
    "claims__definitional_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__logical_gemini_600_full",
    "ethics__commonsense",
    "roleplaying__plain",
    "insider_trading__upscale",
    "sandbagging_v2__wmdp_mmlu",
]

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def make_sgd_classifier():
    """Return an SGDClassifier configured for logistic regression."""
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=4000,
        tol=1e-5,
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )


def train_test_split_pair(X, y):
    """Split data into train/test while keeping paired rows together."""
    n_pairs = len(y) // 2
    X_pairs = X.reshape(n_pairs, 2, -1)
    y_pairs = y.reshape(n_pairs, 2)

    X_pairs_train, X_pairs_test, y_pairs_train, y_pairs_test = train_test_split(
        X_pairs, y_pairs, test_size=0.2, random_state=42
    )

    X_train = X_pairs_train.reshape(-1, X.shape[1])
    X_test = X_pairs_test.reshape(-1, X.shape[1])
    y_train = y_pairs_train.flatten()
    y_test = y_pairs_test.flatten()
    return X_train, X_test, y_train, y_test


def compute_erasers(data, erase_tasks, scaler):
    """Fit LEACE erasers for each task and report before/after AUROC."""
    all_erasers = {}
    for task in erase_tasks:
        print(f"Erasing task: {task}")
        X_train = scaler.transform(data[task]["X_train"])
        X_test = scaler.transform(data[task]["X_test"])
        y_train, y_test = data[task]["y_train"], data[task]["y_test"]

        X_train_torch = torch.from_numpy(X_train).cuda().to(torch.float32)
        y_train_torch = torch.from_numpy(y_train).cuda().to(torch.float32)
        X_test_torch = torch.from_numpy(X_test).cuda().to(torch.float32)

        # Pre-erasure probe
        real_lr = make_sgd_classifier()
        real_lr.fit(X_train, y_train)
        auroc_ori = roc_auc_score(y_test, real_lr.decision_function(X_test))
        print(f"  original auroc on test set: {auroc_ori:.4f}")

        # Fit eraser
        print("  Erasing concept...")
        start_time = time.time()
        eraser = LeaceEraser.fit(X_train_torch, y_train_torch)
        all_erasers[task] = eraser
        print(f"  Concept erased after {(time.time() - start_time):.4f}s!")

        # Post-erasure probe
        X_train_trans = eraser(X_train_torch).cpu().numpy()
        X_test_trans = eraser(X_test_torch).cpu().numpy()

        null_lr = make_sgd_classifier()
        null_lr.fit(X_train_trans, y_train)

        auroc_null = roc_auc_score(y_test, null_lr.decision_function(X_test_trans))
        print(f"  transformed auroc on test set: {auroc_null:.4f}")

        auroc_null_train = roc_auc_score(
            y_train, null_lr.decision_function(X_train_trans)
        )
        print(f"  transformed auroc on train set: {auroc_null_train:.4f}\n")

    return all_erasers


def run_cross_task_evaluation(
    data, all_erasers, all_test_data, scaler, inlp_task_list, train_task_list, test_task_list
):
    """Evaluate original and transformed probes across all task combinations."""
    all_results = {}

    for inlp_task in inlp_task_list:
        print(f"Erase {inlp_task}...")
        all_results[inlp_task] = {}
        eraser = all_erasers[inlp_task]

        for train_task in train_task_list:
            print(f"    Train on {train_task}...")
            all_results[inlp_task][train_task] = {
                "original": {},
                "transformed": {},
            }

            X_train_orig = scaler.transform(data[train_task]["X_train"])
            X_test_orig = scaler.transform(data[train_task]["X_test"])
            y_train = data[train_task]["y_train"]
            y_test = data[train_task]["y_test"]

            # ── Original features ──
            model_orig = make_sgd_classifier()
            model_orig.fit(X_train_orig, y_train)

            test_acc_orig = model_orig.score(X_test_orig, y_test)
            test_auroc_orig = roc_auc_score(
                y_test, model_orig.decision_function(X_test_orig)
            )
            all_results[inlp_task][train_task]["original"][train_task] = {
                "acc": test_acc_orig,
                "auroc": test_auroc_orig,
            }

            # ── Transformed features ──
            X_train_torch = torch.from_numpy(X_train_orig).cuda().to(torch.float32)
            X_test_torch = torch.from_numpy(X_test_orig).cuda().to(torch.float32)
            X_train_trans = eraser(X_train_torch).cpu().numpy()
            X_test_trans = eraser(X_test_torch).cpu().numpy()

            model_trans = make_sgd_classifier()
            model_trans.fit(X_train_trans, y_train)

            test_acc_trans = model_trans.score(X_test_trans, y_test)
            test_auroc_trans = roc_auc_score(
                y_test, model_trans.decision_function(X_test_trans)
            )
            all_results[inlp_task][train_task]["transformed"][train_task] = {
                "acc": test_acc_trans,
                "auroc": test_auroc_trans,
            }

            # ── OOD testing ──
            for test_task in test_task_list:
                if test_task == train_task:
                    continue
                print(f"        Test on {test_task}...")

                X_ood_test, y_ood_test = all_test_data[test_task]
                X_ood_test_orig = scaler.transform(X_ood_test)

                # Original probe on original features
                y_ood_scores_orig = model_orig.decision_function(X_ood_test_orig)
                ood_auroc_orig = roc_auc_score(y_ood_test, y_ood_scores_orig)
                max_acc_orig, _ = utils.compute_max_acc(y_ood_scores_orig, y_ood_test)
                all_results[inlp_task][train_task]["original"][test_task] = {
                    "acc": max_acc_orig,
                    "auroc": ood_auroc_orig,
                }

                # Transformed probe on transformed features
                X_ood_torch = torch.from_numpy(X_ood_test_orig).cuda().to(torch.float32)
                X_ood_trans = eraser(X_ood_torch).cpu().numpy()

                y_ood_scores_trans = model_trans.decision_function(X_ood_trans)
                ood_auroc_trans = roc_auc_score(y_ood_test, y_ood_scores_trans)
                max_acc_trans, _ = utils.compute_max_acc(y_ood_scores_trans, y_ood_test)
                all_results[inlp_task][train_task]["transformed"][test_task] = {
                    "acc": max_acc_trans,
                    "auroc": ood_auroc_trans,
                }

    return all_results


def plot_removed_task_heatmap(
    all_results, inlp_task, train_task_list, test_task_list, metric="auroc"
):
    """
    Plot heatmap for a single removed task showing transformed metric
    across training tasks (rows) and test tasks (columns).
    """
    xticklabels = [CUSTOM_LABELS[t] for t in test_task_list]
    yticklabels = [CUSTOM_LABELS[t] for t in train_task_list]

    data = np.zeros((len(train_task_list), len(test_task_list)))
    for i, train_task in enumerate(train_task_list):
        for j, test_task in enumerate(test_task_list):
            if test_task in all_results[inlp_task][train_task]["transformed"]:
                data[i, j] = all_results[inlp_task][train_task]["transformed"][
                    test_task
                ][metric]
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=0.5,
        vmax=1.0,
        ax=ax,
        annot_kws={"size": 12},
        cbar_kws={"label": metric.upper()},
    )

    # Highlight train==test cells
    for i, train_task in enumerate(train_task_list):
        for j, test_task in enumerate(test_task_list):
            if train_task == test_task:
                rect = Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="red", linewidth=3
                )
                ax.add_patch(rect)

    # Highlight erased task column
    if inlp_task in test_task_list:
        idx = test_task_list.index(inlp_task)
        rect = Rectangle(
            (idx, 0),
            1,
            len(train_task_list),
            fill=False,
            edgecolor="limegreen",
            linewidth=3,
            linestyle="--",
        )
        ax.add_patch(rect)

    ax.set_title(
        f"LEACE Removed Task: {CUSTOM_LABELS[inlp_task]}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Test Task", fontsize=14)
    ax.set_ylabel("Train Task", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), size=12, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), size=12, rotation=0)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # Load features
    print("Loading extracted features...")
    extracted_feats = h5py.File(FEATS_PATH, "r")
    print(f"  Feature keys: {list(extracted_feats.keys())}")

    # Load tokenizer
    print("Loading tokenizer...")
    _, tokenizer = get_model_and_tokenizer(
        MODEL_NAME,
        models_directory="../deception-detection/data/huggingface/",
        omit_model=True,
    )

    # Load datasets
    print("Loading datasets...")
    task_list = config.TASK_LISTS["default"]["test"]
    all_datasets = utils.get_all_dataset(task_list, MODEL_NAME)
    print(f"  Dataset keys: {list(all_datasets.keys())}")

    # Prepare train/test splits (pair-aware)
    print("Preparing pair-aware train/test splits...")
    data = {}
    for task in ERASE_TASKS:
        print(f"  {task}")
        prepared = utils.prepare_data(
            task,
            LAYER_NAME,
            extracted_feats,
            all_datasets,
            tokenizer,
            feature_type=FEATURE_TYPE,
            balance_groups=False,
        )
        X_train, X_test, y_train, y_test = train_test_split_pair(prepared.X, prepared.y)
        data[task] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    # Fit global scaler
    print("Fitting global StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(np.concatenate([data[t]["X_train"] for t in ERASE_TASKS]))

    # Compute LEACE erasers
    print("\n=== Computing LEACE erasers ===")
    all_erasers = compute_erasers(data, ERASE_TASKS, scaler)

    # Prepare OOD test data
    print("\nPreparing OOD test data...")
    all_test_data = {}
    for test_task in TEST_TASKS:
        prepared = utils.prepare_data(
            test_task,
            LAYER_NAME,
            extracted_feats,
            all_datasets,
            tokenizer,
            feature_type=FEATURE_TYPE,
            balance_groups=False,
        )
        all_test_data[test_task] = (prepared.X, prepared.y)

    # Run cross-task evaluation
    print("\n=== Running cross-task evaluation ===")
    all_results = run_cross_task_evaluation(
        data, all_erasers, all_test_data, scaler, ERASE_TASKS, TRAIN_TASKS, TEST_TASKS
    )

    # Save results
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.save(SAVE_PATH, all_results, allow_pickle=True)
    print(f"\nResults saved to {SAVE_PATH}")

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_style("whitegrid")

    for inlp_task in PLOT_INLP_TASKS:
        fig = plot_removed_task_heatmap(
            all_results, inlp_task, PLOT_TRAIN_TASKS, PLOT_TEST_TASKS, metric="auroc"
        )
        save_name = os.path.join(PLOT_DIR, f"leace_removed_heatmap_{inlp_task}.pdf")
        fig.savefig(save_name, dpi=300)
        print(f"  Saved {save_name}")
        plt.close(fig)

    extracted_feats.close()
    print("Done.")


if __name__ == "__main__":
    main()