"""Quick comparison: sklearn LogisticRegression vs SGDClassifier on FLEED claims."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from dotenv import load_dotenv
load_dotenv()

from src import utils
from src.datasets import DialogueDataset
from src.models import get_model_and_tokenizer

TASKS = [
    'claims__definitional_gemini_600_full',
    'claims__evidential_gemini_600_full',
    'claims__fictional_gemini_600_full',
    'claims__logical_gemini_600_full',
]

def get_data(task, feats, all_datasets, tokenizer):
    prepared = utils.prepare_data(
        task, 'layer_13', feats, all_datasets,
        tokenizer, balance_groups=False, feature_type='average'
    )
    return prepared.X, prepared.y

def eval_cv(X, y, make_model, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs, aucs = [], []
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = make_model()
        model.fit(X_tr, y[train_idx])
        preds = model.predict(X_te)
        scores = model.decision_function(X_te)
        accs.append(accuracy_score(y[test_idx], preds))
        aucs.append(roc_auc_score(y[test_idx], scores))
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)

def make_sgd():
    return SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-4,
        max_iter=50000, tol=1e-5, n_jobs=-1, random_state=42
    )

def make_lr():
    # alpha=1e-4 in SGD ~ C=1/alpha but not exactly due to SGD vs exact solver
    # We'll test a few C values
    return LogisticRegression(
        penalty='l2', C=1e4, max_iter=10000, solver='lbfgs', random_state=42
    )

def make_lr_c1():
    return LogisticRegression(
        penalty='l2', C=1.0, max_iter=10000, solver='lbfgs', random_state=42
    )

def main():
    feats = h5py.File('./results/extracted_feats_layer13_llama-8b.h5', 'r')
    _, tokenizer = get_model_and_tokenizer('llama-8b', './data/huggingface/', omit_model=True)
    all_datasets = utils.get_all_dataset(TASKS, 'llama-8b')

    print(f"{'Task':<45} {'Model':<20} {'Acc':>10} {'AUROC':>10}")
    print("-" * 90)

    for task in TASKS:
        X, y = get_data(task, feats, all_datasets, tokenizer)
        short = task.split('__')[1].replace('_gemini_600_full', '')

        for name, factory in [('SGDClassifier', make_sgd), ('LR (C=1e4)', make_lr), ('LR (C=1)', make_lr_c1)]:
            acc_m, acc_s, auc_m, auc_s = eval_cv(X, y, factory)
            print(f"{short:<45} {name:<20} {acc_m:.4f}±{acc_s:.4f} {auc_m:.4f}±{auc_s:.4f}")
        print()

    feats.close()

if __name__ == '__main__':
    main()
