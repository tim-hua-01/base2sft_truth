"""Sweep LogisticRegression C values: in-distribution CV + OOD cross-domain generalization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from dotenv import load_dotenv
load_dotenv()

from src import utils
from src.models import get_model_and_tokenizer

TASKS = [
    'claims__definitional_gemini_600_full',
    'claims__evidential_gemini_600_full',
    'claims__fictional_gemini_600_full',
    'claims__logical_gemini_600_full',
    'got__best',
]
SHORT = {
    'claims__definitional_gemini_600_full': 'definitional',
    'claims__evidential_gemini_600_full': 'evidential',
    'claims__fictional_gemini_600_full': 'fictional',
    'claims__logical_gemini_600_full': 'logical',
    'got__best': 'GoT',
}

LAYER = 15
C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 1e4]


def get_data(task, feats, all_datasets, tokenizer):
    prepared = utils.prepare_data(
        task, f'layer_{LAYER}', feats, all_datasets,
        tokenizer, balance_groups=False, feature_type='average'
    )
    return prepared.X, prepared.y


def eval_in_distribution(X, y, C, n_folds=5):
    """5-fold CV on a single dataset. Returns (mean_acc, std_acc, mean_auc, std_auc)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs, aucs = [], []
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = LogisticRegression(C=C, max_iter=10000, solver='lbfgs', random_state=42)
        model.fit(X_tr, y[train_idx])
        scores = model.decision_function(X_te)
        accs.append(accuracy_score(y[test_idx], model.predict(X_te)))
        aucs.append(roc_auc_score(y[test_idx], scores))
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)


def eval_ood(X_train, y_train, X_test, y_test, C):
    """Train on full dataset A, test on full dataset B. Returns (acc, auc)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = LogisticRegression(C=C, max_iter=10000, solver='lbfgs', random_state=42)
    model.fit(X_tr, y_train)
    scores = model.decision_function(X_te)
    acc = accuracy_score(y_test, model.predict(X_te))
    auc = roc_auc_score(y_test, scores)
    return acc, auc


def main():
    import warnings
    warnings.filterwarnings('ignore')

    feats = h5py.File(f'./results/extracted_feats_layer{LAYER}_llama-8b.h5', 'r')
    _, tokenizer = get_model_and_tokenizer('llama-8b', './data/huggingface/', omit_model=True)
    all_datasets = utils.get_all_dataset(TASKS, 'llama-8b')

    # Load all data
    data = {}
    for task in TASKS:
        X, y = get_data(task, feats, all_datasets, tokenizer)
        data[task] = (X, y)
        print(f"Loaded {SHORT[task]}: {X.shape[0]} samples")
    print()

    # === Part 1: In-distribution CV sweep ===
    print("=" * 100)
    print("PART 1: IN-DISTRIBUTION CV (5-fold)")
    print("=" * 100)

    header = f"{'C':>10}"
    for t in TASKS:
        header += f"  {SHORT[t]+' acc':>16} {SHORT[t]+' auc':>16}"
    print(header)
    print("-" * len(header))

    id_results = {}
    for C in C_VALUES:
        row = f"{C:>10.0e}" if C >= 100 else f"{C:>10g}"
        for task in TASKS:
            X, y = data[task]
            acc_m, acc_s, auc_m, auc_s = eval_in_distribution(X, y, C)
            row += f"  {acc_m:.4f}±{acc_s:.4f} {auc_m:.4f}±{auc_s:.4f}"
            id_results[(C, task)] = (acc_m, acc_s, auc_m, auc_s)
        print(row)
    print()

    # === Part 2: OOD cross-domain sweep ===
    print("=" * 100)
    print("PART 2: OOD CROSS-DOMAIN GENERALIZATION (train on A, test on B)")
    print("=" * 100)

    for C in C_VALUES:
        print(f"\n--- C = {C} ---")
        label = 'train \\ test'
        header = f"{label:<15}"
        for t in TASKS:
            header += f" {SHORT[t]:>12}"
        print(header)
        print("-" * len(header))

        for train_task in TASKS:
            X_tr, y_tr = data[train_task]
            row = f"{SHORT[train_task]:<15}"
            for test_task in TASKS:
                if train_task == test_task:
                    # Use CV result for diagonal
                    _, _, auc_m, _ = id_results[(C, train_task)]
                    row += f" {auc_m:>11.4f}*"
                else:
                    X_te, y_te = data[test_task]
                    _, auc = eval_ood(X_tr, y_tr, X_te, y_te, C)
                    row += f" {auc:>12.4f}"
            print(row)

    feats.close()


if __name__ == '__main__':
    main()
