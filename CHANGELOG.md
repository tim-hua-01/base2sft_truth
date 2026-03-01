# Changelog

## Sycophancy Dataset V2 Pipeline

### Added `scripts/create_sycophancy_dataset_v2.py`
- Clean rewrite of the sycophancy generation pipeline with two separated stages:
  - `generate`: pre-filters questions by base + instruct model confidence, builds all prompts upfront, runs batched HuggingFace inference (no nnsight), saves raw JSON with full-vocab log_softmax and 4-letter softmax probs
  - `pair`: loads raw JSON, validates flips using log-prob shift thresholds (not just argmax), picks one pair per question, saves wide CSV with diagnostic columns
- Bio templates loaded from JSON with `{question_subject}` and `{answer_letter}` placeholders for flexible phrasing
- Supports `--sample-n` for quick test runs and `--tasks` for subset selection
- Pair output filenames include timestamp by default (`--output-name` to override)

### Added `scripts/analyze_bio_sycophancy.py`
- Ranks bio templates by sycophancy effectiveness
- Computes per-bio shift: `(log P(hinted) - log P(correct))_bio - (log P(hinted) - log P(correct))_baseline`
- Outputs ranked table with mean/median shift, flip rate, and sample counts

### Added bio template files
- `data/sycophancy_v2/bio_templates.json`: 36 templates across 9 categories (credential_authority, casual_baseline, self_asserted_competence, rhetorical_pressure, emotional_vulnerability, emotional_enthusiasm, social_proof, confidence_dominance, stakes_urgency)
- `data/sycophancy_v2/bio_templates_llama_8b.json`: top 6 most effective templates for Llama-3.1-8B-Instruct (stakes_urgency, confidence_dominance, self_asserted_competence, credential_authority)

## Changes from Original truth_spec Codebase

### Filter MMLU questions based on base model 5-shot performance
- Added `scripts/measure_base_confidence.py` to run 5-shot MMLU STEM evaluation on base models and record per-question P(correct)
- Allows downstream filtering of questions where the base model is already confident (e.g. P(correct) > 0.75), so sycophancy experiments only use questions where the model is uncertain
- Output dir defaults to `./data/sycophancy_v2/<model>/base_confidence/` (model-namespaced)

### Measure instruct model confidence with chat-template prompting
- Added `scripts/measure_instruct_confidence.py` to evaluate instruct models on MMLU STEM using the same prefilled assistant turn (`"I believe the best answer is ("`) as the sycophancy pipeline
- Supports 0-shot and few-shot (1–5) evaluation via `--n-shot`; few-shot examples are injected into the user message
- Saves per-question P(correct) and P(A/B/C/D) to `./data/sycophancy_v2/<model>/instruct_confidence/<model>_<n>shot.csv`
- Confidence scores are directly comparable to base model scores since both extract next-token softmax probabilities over {A, B, C, D}
- Output dir defaults to `./data/sycophancy_v2/<model>/instruct_confidence/` (model-namespaced)

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
