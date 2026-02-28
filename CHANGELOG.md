# Changelog

## Changes from Original truth_spec Codebase

### Switch from SGDClassifier to LogisticRegression (C=0.001)
- **Original**: `SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4)` with post-hoc weight normalization (`coef_ /= norm`)
- **New**: `LogisticRegression(C=0.001, solver='lbfgs', max_iter=10000)`
- **Reason**: Two issues with the original:
  1. SGDClassifier had high variance in accuracy across CV folds (e.g., ±0.045 on evidential vs ±0.008 with LR) due to stochastic convergence
  2. The effective regularization was far too weak (alpha=1e-4 ≈ C=1e4) for the high-dimensional regime (~4096 features, ~1150 samples)
- **Hyperparameter sweep** (`scripts/sweep_lr_regularization.py`) over C ∈ {0.001, 0.01, 0.1, 1.0, 1e4} showed C=0.001 is optimal:
  - In-distribution: logical AUROC 0.785 → 0.844 (+0.059)
  - OOD: logical→fictional 0.878 → 0.957 (+0.079), logical→evidential 0.957 → 0.995 (+0.038)
  - Easy tasks (definitional, evidential, fictional) barely affected
- **Note**: The `regularization` CLI arg now maps to C directly (not alpha). Pass `--regularization 0.001` for the recommended setting.
- See `scripts/compare_lr_vs_sgd.py` and `scripts/sweep_lr_regularization.py` for full results.

### Added python-dotenv for HF_TOKEN loading
- Added `load_dotenv()` to `scripts/extract_activation.py` and `scripts/train_test_probes.py`
- HF token loaded from `.env` file automatically

### Removed tracked cache files
- Removed 57 `__pycache__` and `.ipynb_checkpoints` files from git tracking (already in `.gitignore`)
