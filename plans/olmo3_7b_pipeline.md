# OLMo 3 7B: Full Sycophancy + Probe Pipeline

## Context
Replicate the full llama-8b pipeline for OLMo 3 7B: bio ranking → sycophancy generation → activation extraction (GOT+FLEED+sycophancy_v2) → probe training + transfer matrix → plots. Both instruct (`olmo3-7b`) and base (`olmo3-7b-purebase`, off-policy).

**Key differences from llama-8b run:**
- Layers 10-25 (not 10-20) for extraction
- Layer sweep over 10-25 to pick best layer (by FLEED in-distribution AUROC), then full transfer matrix at that layer
- Bio ranking run to select OLMo-specific top-12 bios (not reusing llama rankings)

## Prerequisites (already done)
- Confidence CSVs already exist:
  - `data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv` (4097 questions)
  - `data/sycophancy_v2/olmo3-7b-purebase/base_confidence/olmo3-7b-purebase_5shot.csv` (4097 questions)
- Model registered in `src/models.py`:
  - `olmo3-7b` → `allenai/Olmo-3-7B-Instruct`
  - `olmo3-7b-purebase` → `allenai/Olmo-3-1025-7B` (revision `stage1-step1413814`)

---

## Step 0: Fix MODEL_LAYER_CONFIGS in extract_activation.py

**File:** `scripts/extract_activation.py` (lines 52-58)

**Problem:** The current config has `'7b': {'num_layers': 28}` (for Qwen 7B). Model name matching iterates keys and checks `if key in model_name`. For `olmo3-7b`, `'7b'` matches → 28 layers (wrong). OLMo 3 7B actually has **32 layers**.

**Fix:** Add `'olmo3-7b': {'num_layers': 32}` entry before `'7b'` so more-specific keys match first.

---

## Step 1: Bio ranking run (60q × 36 bios)

**Purpose:** Identify which of the 36 bio templates are most effective at shifting OLMo 3 7B toward sycophantic answers.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/create_sycophancy_dataset_v2.py generate \
    --model olmo3-7b \
    --base-confidence-csv data/sycophancy_v2/olmo3-7b-purebase/base_confidence/olmo3-7b-purebase_5shot.csv \
    --instruct-confidence-csv data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv \
    --bio-templates data/sycophancy_v2/bio_templates.json \
    --output-dir data/sycophancy_v2 \
    --batch-size 64 \
    --sample-n 60 \
    --seed 42
```

**Expected output:** `data/sycophancy_v2/olmo3-7b/raw/raw_60q_36bios_seed42.json`

---

## Step 2: Analyze bio sycophancy & create flip stats file

### 2a: Run analysis
```bash
uv run python scripts/analyze_bio_sycophancy.py \
    --raw-path data/sycophancy_v2/olmo3-7b/raw/raw_60q_36bios_seed42.json \
    --instruct-confidence-csv data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv \
    | tee data/sycophancy_v2/olmo3-7b/bio_sycophancy_analysis.txt
```

### 2b: Create detailed flip stats CSV
Per-bio CSV with columns: `bio_id, category, mean_syco_shift, median_syco_shift, flip_rate, n_combos, n_flips`

### 2c: Select top 12 bios → create bio_templates_olmo3_7b.json

---

## Step 3: Production sycophancy generation (all questions × 12 bios, bpq=6)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/create_sycophancy_dataset_v2.py generate \
    --model olmo3-7b \
    --base-confidence-csv data/sycophancy_v2/olmo3-7b-purebase/base_confidence/olmo3-7b-purebase_5shot.csv \
    --instruct-confidence-csv data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv \
    --bio-templates data/sycophancy_v2/bio_templates_olmo3_7b.json \
    --output-dir data/sycophancy_v2 \
    --batch-size 64 \
    --max-bios-per-question 6 \
    --seed 123172
```

---

## Step 4: Pair into contrastive dataset

```bash
uv run python scripts/create_sycophancy_dataset_v2.py pair \
    --model olmo3-7b \
    --instruct-confidence-csv data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv \
    --raw-path data/sycophancy_v2/olmo3-7b/raw/<production_raw_file>.json \
    --output-dir data/sycophancy_v2 \
    --max-pairs-per-question 2 \
    --seed 123172
```

---

## Step 5: Extract activations — olmo3-7b instruct, layers 10-25

### 5a: GOT + FLEED
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b \
    --layers 10-25 \
    --tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
            claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best \
    --output ./results/activations_got_plus_fleed_olmo3-7b_layers10-25.h5
```

### 5b: Sycophancy V2
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b \
    --layers 10-25 \
    --tasks "sycophancy_v2__data/sycophancy_v2/olmo3-7b/pairs/<pairs_file>.csv" \
    --output ./results/activations_sycophancy_v2_olmo3-7b_layers10-25.h5
```

### 5c: Merge sycophancy_v2 into GOT+FLEED HDF5

---

## Step 6: Extract activations — olmo3-7b-purebase (off-policy), layers 10-25

Same as Step 5 but with `--model olmo3-7b-purebase`.

---

## Step 7: Layer sweep — find best layer by FLEED in-distribution AUROC

```bash
uv run python scripts/train_test_probes.py \
    --model-name olmo3-7b \
    --features-file ./results/activations_got_plus_fleed_olmo3-7b_layers10-25.h5 \
    --layer-idx 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full \
    --results-csv ./results/layer_sweep_olmo3-7b_fleed.csv \
    --force-retrain --verbose
```

Pick best layer L by mean AUROC across the 4 FLEED tasks (cv results).

---

## Step 8: Full 6×6 transfer matrix at best layer

### 8a: olmo3-7b instruct at layer L
### 8b: olmo3-7b-purebase at layer L (same L used for both)

---

## Step 9: Generate plots

- Side-by-side instruct vs base AUROC heatmaps
- Delta heatmap (instruct − base)
- Individual heatmaps with diagonal borders
- Layer sweep line plot (FLEED AUROC vs layer)

---

## Step 10: Update commands_run.md

Log all commands with output filenames, stats, and runtime notes.
