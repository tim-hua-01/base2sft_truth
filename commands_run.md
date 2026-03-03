# Commands Run

## Notes

- `data/sycophancy_v2/llama-8b/raw/raw_60q_36bios_seed42_bio_ranking_run.json` (8640 records, 60 questions × 36 bios × 4 answers) is from the bio template ranking run using all 36 bio templates on 60 sampled questions. This was the run used to produce `data/sycophancy_v2/llama-8b/bio_sycophancy_analysis.txt` via `analyze_bio_sycophancy.py`. It is NOT a production sycophancy dataset — it was used to rank bios by mean sycophancy shift to select the top-performing templates.

## Sycophancy V2 Commands

### Full generation: llama-8b with top-12 bios, bpq=6 (2026-03-02)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/create_sycophancy_dataset_v2.py generate \
    --model llama-8b \
    --base-confidence-csv data/sycophancy_v2/llama-8b-base/base_confidence/llama-8b-base.csv \
    --instruct-confidence-csv data/sycophancy_v2/llama-8b/instruct_confidence/llama-8b_0shot.csv \
    --bio-templates data/sycophancy_v2/bio_templates_llama_8b.json \
    --output-dir data/sycophancy_v2 \
    --batch-size 64 \
    --max-bios-per-question 6 \
    --seed 123172

# Output: data/sycophancy_v2/llama-8b/raw/raw_1350q_12bios_bpq6_seed123172.json
# 32400 records, 1350 questions × 6 bios/question (from 12) × 4 answers
# Runtime: ~12.5 min on L40S (46GB), batch_size=64
# Note: batch_size=128 and 256 OOM'd due to variable sequence lengths across batches

uv run python scripts/create_sycophancy_dataset_v2.py pair \
    --model llama-8b \
    --instruct-confidence-csv data/sycophancy_v2/llama-8b/instruct_confidence/llama-8b_0shot.csv \
    --raw-path data/sycophancy_v2/llama-8b/raw/raw_1350q_12bios_bpq6_seed123172.json \
    --output-dir data/sycophancy_v2 \
    --seed 123172

# Output: data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_05:34:21.csv
# 470 pairs from 1350 questions (34.8% flip rate)
# Mean P(correct) honest: 0.961, syco: 0.295
# Mean logprob_shift_correct: -1.445, logprob_shift_syco_ans: +5.208
# 24 STEM subjects represented
```

### Re-pair with max 2 pairs per question (2026-03-02)

```bash
uv run python scripts/create_sycophancy_dataset_v2.py pair \
    --model llama-8b \
    --instruct-confidence-csv data/sycophancy_v2/llama-8b/instruct_confidence/llama-8b_0shot.csv \
    --raw-path data/sycophancy_v2/llama-8b/raw/raw_1350q_12bios_bpq6_seed123172.json \
    --output-dir data/sycophancy_v2 \
    --max-pairs-per-question 2 \
    --seed 123172

# Output: data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv
# 834 pairs from 470 questions (1.8 per question avg — some questions had only 1 valid combo)
# Mean P(correct) honest: 0.956, syco: 0.276
# Mean logprob_shift_correct: -1.555, logprob_shift_syco_ans: +5.176
# 24 STEM subjects represented
```

## Activation Extraction Commands

### GOT + FLEED: llama-8b, layers 10-20 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model llama-8b \
    --layers 10-20 \
    --tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
            claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best \
    --output ./results/activations_got_plus_fleed_llama-8b_layers10-20.h5

# Output: results/activations_got_plus_fleed_llama-8b_layers10-20.h5
# 5 tasks, 11 layers (10-20), batch_size=16
# got__best: 6952 masked tokens
# ~7.5 batch/s per task on L40S (46GB)
```

### Sycophancy V2: llama-8b, layers 10-20 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model llama-8b \
    --layers 10-20 \
    --tasks "sycophancy_v2__data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv" \
    --output ./results/activations_sycophancy_v2_llama-8b_layers10-20.h5

# Output: results/activations_sycophancy_v2_llama-8b_layers10-20.h5
# 1668 dialogues (834 pairs), 11 layers (10-20), batch_size=16
# ~3.0s/batch on L40S (46GB), ~5.3 min total extraction
```

### GOT + FLEED: llama-8b-base, layers 10-20 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model llama-8b-base \
    --layers 10-20 \
    --tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
            claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best \
    --output ./results/activations_got_plus_fleed_llama-8b-base_layers10-20.h5

# Output: results/activations_got_plus_fleed_llama-8b-base_layers10-20.h5
```

### Sycophancy V2: llama-8b-base (off-policy), layers 10-20 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model llama-8b-base \
    --layers 10-20 \
    --tasks "sycophancy_v2__data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv" \
    --output ./results/activations_sycophancy_v2_llama-8b-base_layers10-20.h5

# Output: results/activations_sycophancy_v2_llama-8b-base_layers10-20.h5
# Off-policy: uses instruct-generated sycophancy pairs with base model activations
# 1668 dialogues (834 pairs), 11 layers (10-20), batch_size=16
```

Note: sycophancy_v2 activations were merged into the got_plus_fleed HDF5 files for both models using h5py copy.

## Probe Training & Transfer Commands

### Transfer matrix: llama-8b instruct, layer 15 (2026-03-03)

```bash
SYCO_TASK="sycophancy_v2__data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv"

uv run python scripts/train_test_probes.py \
    --model-name llama-8b \
    --features-file ./results/activations_got_plus_fleed_llama-8b_layers10-20.h5 \
    --layer-idx 15 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --results-csv ./results/transfer_llama-8b_layer15.csv \
    --force-retrain --verbose

# Output: results/transfer_llama-8b_layer15.csv
# 6×6 transfer matrix (AUROC, max_acc, etc.)
# Probes: results/probes/llama-8b/{task}_last/
```

### Transfer matrix: llama-8b-base, layer 15 (2026-03-03)

```bash
uv run python scripts/train_test_probes.py \
    --model-name llama-8b-base \
    --features-file ./results/activations_got_plus_fleed_llama-8b-base_layers10-20.h5 \
    --layer-idx 15 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --results-csv ./results/transfer_llama-8b-base_layer15.csv \
    --force-retrain --verbose

# Output: results/transfer_llama-8b-base_layer15.csv
# Off-policy sycophancy: instruct-generated pairs, base model activations
```

---

## OLMo 3 7B Pipeline (2026-03-03)

### Notes

- OLMo 3 7B has 32 layers (not 28 like Qwen 7B). Added `'olmo3-7b': {'num_layers': 32}` to `MODEL_LAYER_CONFIGS` in `extract_activation.py`.
- Layers 10-25 used for extraction (vs 10-20 for llama-8b).
- Layer sweep picked **layer 18** as best (mean FLEED CV AUROC = 0.9733). Same layer used for both instruct and base.
- Bio ranking identified different top-12 bios than llama-8b (e.g., `confidence_dominance_1` #1 for OLMo vs `stakes_urgency_2` for llama).

### Bio ranking run: olmo3-7b, 60q × 36 bios (2026-03-03)

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

# Output: data/sycophancy_v2/olmo3-7b/raw/raw_60q_36bios_seed42.json
# 8640 records (60 questions × 36 bios × 4 answers)
# Intersection of confident questions: 1002 (from 1244 base + 2060 instruct)
# Runtime: ~5 min on L40S (46GB)
```

Analysis: `data/sycophancy_v2/olmo3-7b/bio_sycophancy_analysis.txt`
- Top bio: `confidence_dominance_1` (shift=+7.97, flip=18.3%)
- Overall mean sycophancy shift: +4.90, flip rate: 7.2%
- OLMo top-12 bios saved to `data/sycophancy_v2/bio_templates_olmo3_7b.json`
- Per-bio stats: `data/sycophancy_v2/olmo3-7b/bio_flip_stats.csv`

### Full generation: olmo3-7b with top-12 bios, bpq=6 (2026-03-03)

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

# Output: data/sycophancy_v2/olmo3-7b/raw/raw_1002q_12bios_bpq6_seed123172.json
# 24048 records (1002 questions × 6 bios/question × 4 answers)
# Runtime: ~10 min on L40S (46GB)
```

### Pair with max 2 pairs per question (2026-03-03)

```bash
uv run python scripts/create_sycophancy_dataset_v2.py pair \
    --model olmo3-7b \
    --instruct-confidence-csv data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv \
    --raw-path data/sycophancy_v2/olmo3-7b/raw/raw_1002q_12bios_bpq6_seed123172.json \
    --output-dir data/sycophancy_v2 \
    --max-pairs-per-question 2 \
    --seed 123172

# Output: data/sycophancy_v2/olmo3-7b/pairs/sycophancy_pairs_26-03-03_02:24:05.csv
# 742 pairs from 421 questions (42.0% flip rate — higher than llama's 34.8%)
# Mean P(correct) honest: 0.934, syco: 0.174
# Mean logprob_shift_correct: -2.554, logprob_shift_syco_ans: +6.986
# 24 STEM subjects represented
```

### GOT + FLEED: olmo3-7b instruct, layers 10-25 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b \
    --layers 10-25 \
    --tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
            claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best \
    --output ./results/activations_got_plus_fleed_olmo3-7b_layers10-25.h5

# Output: results/activations_got_plus_fleed_olmo3-7b_layers10-25.h5
# 5 tasks, 16 layers (10-25), batch_size=16
```

### Sycophancy V2: olmo3-7b instruct, layers 10-25 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b \
    --layers 10-25 \
    --tasks "sycophancy_v2__data/sycophancy_v2/olmo3-7b/pairs/sycophancy_pairs_26-03-03_02:24:05.csv" \
    --output ./results/activations_sycophancy_v2_olmo3-7b_layers10-25.h5

# Output: results/activations_sycophancy_v2_olmo3-7b_layers10-25.h5
# 1484 dialogues (742 pairs), 16 layers (10-25), batch_size=16
```

### GOT + FLEED: olmo3-7b-purebase, layers 10-25 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b-purebase \
    --layers 10-25 \
    --tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
            claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best \
    --output ./results/activations_got_plus_fleed_olmo3-7b-purebase_layers10-25.h5

# Output: results/activations_got_plus_fleed_olmo3-7b-purebase_layers10-25.h5
# 5 tasks, 16 layers (10-25), batch_size=16
```

### Sycophancy V2: olmo3-7b-purebase (off-policy), layers 10-25 (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b-purebase \
    --layers 10-25 \
    --tasks "sycophancy_v2__data/sycophancy_v2/olmo3-7b/pairs/sycophancy_pairs_26-03-03_02:24:05.csv" \
    --output ./results/activations_sycophancy_v2_olmo3-7b-purebase_layers10-25.h5

# Output: results/activations_sycophancy_v2_olmo3-7b-purebase_layers10-25.h5
# Off-policy: uses instruct-generated sycophancy pairs with base model activations
# 1484 dialogues (742 pairs), 16 layers (10-25), batch_size=16
```

Note: sycophancy_v2 activations were merged into the got_plus_fleed HDF5 files for both models using h5py copy.

### Layer sweep: olmo3-7b instruct, FLEED in-distribution (2026-03-03)

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

# Output: results/layer_sweep_olmo3-7b_fleed.csv
# Best layer by mean FLEED CV AUROC: Layer 18 (0.9733)
# Layer 17: 0.9704, Layer 19: 0.9718, Layer 16: 0.9701
# Plateau around layers 15-20, peak at 18
```

### Transfer matrix: olmo3-7b instruct, layer 18 (2026-03-03)

```bash
SYCO_TASK="sycophancy_v2__data/sycophancy_v2/olmo3-7b/pairs/sycophancy_pairs_26-03-03_02:24:05.csv"

uv run python scripts/train_test_probes.py \
    --model-name olmo3-7b \
    --features-file ./results/activations_got_plus_fleed_olmo3-7b_layers10-25.h5 \
    --layer-idx 18 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --results-csv ./results/transfer_olmo3-7b_layer18.csv \
    --force-retrain --verbose

# Output: results/transfer_olmo3-7b_layer18.csv
# 6×6 transfer matrix
# Diagonal CV AUROC: Def=0.989, Evi=0.992, Fic=0.973, Log=0.940, GoT=0.997, Syco=0.940
# FLEED→Syco transfer: 0.72-0.77 range
# Syco→FLEED transfer: 0.65-0.92 range
```

### Transfer matrix: olmo3-7b-purebase, layer 18 (2026-03-03)

```bash
uv run python scripts/train_test_probes.py \
    --model-name olmo3-7b-purebase \
    --features-file ./results/activations_got_plus_fleed_olmo3-7b-purebase_layers10-25.h5 \
    --layer-idx 18 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_TASK" \
    --results-csv ./results/transfer_olmo3-7b-purebase_layer18.csv \
    --force-retrain --verbose

# Output: results/transfer_olmo3-7b-purebase_layer18.csv
# Off-policy sycophancy: instruct-generated pairs, base model activations
# Diagonal CV AUROC: Def=0.996, Evi=0.991, Fic=0.972, Log=0.936, GoT=0.999, Syco=0.961
# Notable: base sycophancy in-distribution AUROC (0.961) higher than instruct (0.940)
```

### Plots (2026-03-03)

- `plots/transfer_auroc_instruct_vs_base_olmo3-7b_layer18.png` — side-by-side heatmaps
- `plots/transfer_auroc_delta_instruct_minus_base_olmo3-7b_layer18.png` — delta heatmap
- `plots/transfer_auroc_olmo3-7b_layer18.png` — instruct individual heatmap
- `plots/transfer_auroc_olmo3-7b-purebase_layer18.png` — base individual heatmap
- `plots/layer_sweep_fleed_olmo3-7b.png` — FLEED AUROC by layer

---

## On-Policy Sycophancy (OLMo 3 7B) (2026-03-03)

### Notes

- **On-policy sycophancy**: pairs that flip BOTH the instruct model AND the base model (vs off-policy = instruct-only flips used with base model activations)
- Created `scripts/validate_sycophancy_on_base.py` with two subcommands: `run` (GPU inference) and `pair` (CPU-only re-pairing)
- Added `--prob-threshold` to both `create_sycophancy_dataset_v2.py` and `validate_sycophancy_on_base.py` for optional P(correct) > threshold checks
- Base model is far more sycophantic: 14,672 base-only flips vs 40 instruct-only, 97.7% of instruct flips also flip the base

### Generate base model raw data + pair (2026-03-03)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/validate_sycophancy_on_base.py run \
    --model olmo3-7b \
    --instruct-raw-path data/sycophancy_v2/olmo3-7b/raw/raw_1002q_12bios_bpq6_seed123172.json \
    --instruct-confidence-csv data/sycophancy_v2/olmo3-7b/instruct_confidence/olmo3-7b_0shot.csv \
    --bio-templates data/sycophancy_v2/bio_templates_olmo3_7b.json \
    --output-dir data/sycophancy_v2 \
    --batch-size 32 \
    --max-pairs-per-question 2 \
    --prob-threshold 0.5 \
    --seed 123172

# Output:
#   data/sycophancy_v2/olmo3-7b-purebase/raw/base_revalidation_26-03-03_05:17:22.json (24048 records)
#   data/sycophancy_v2/olmo3-7b-purebase/pairs/sycophancy_pairs_onpolicy_26-03-03_05:25:36.csv
# 695 on-policy pairs from 414 questions (with --prob-threshold 0.5)
# (731 pairs without prob_threshold, filtering removed 36 pairs = 4.9%)
# Runtime: ~8.5 min on L40S (46GB), batch_size=32 (64 OOM'd)
```

### Activation extraction: on-policy sycophancy, layer 18 only (2026-03-03)

```bash
ONPOLICY_CSV="data/sycophancy_v2/olmo3-7b-purebase/pairs/sycophancy_pairs_onpolicy_26-03-03_05:25:36.csv"

# Instruct model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b --layers 18-18 \
    --tasks "sycophancy_v2__${ONPOLICY_CSV}" \
    --output ./results/activations_sycophancy_v2_onpolicy_olmo3-7b_layer18.h5

# Base model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/extract_activation.py \
    --model olmo3-7b-purebase --layers 18-18 \
    --tasks "sycophancy_v2__${ONPOLICY_CSV}" \
    --output ./results/activations_sycophancy_v2_onpolicy_olmo3-7b-purebase_layer18.h5

# Output: 1390 dialogues (695 pairs) each, layer 18 only
# Merged into main GOT+FLEED HDF5 files using h5py deep copy
```

### 7×7 transfer matrix with on-policy sycophancy, layer 18 (2026-03-03)

```bash
SYCO_OFFPOLICY="sycophancy_v2__data/sycophancy_v2/olmo3-7b/pairs/sycophancy_pairs_26-03-03_02:24:05.csv"
SYCO_ONPOLICY="sycophancy_v2__data/sycophancy_v2/olmo3-7b-purebase/pairs/sycophancy_pairs_onpolicy_26-03-03_05:25:36.csv"

# Instruct model (reuses existing 6 probes, only trains on-policy probe)
uv run python scripts/train_test_probes.py \
    --model-name olmo3-7b \
    --features-file ./results/activations_got_plus_fleed_olmo3-7b_layers10-25.h5 \
    --layer-idx 18 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_OFFPOLICY" "$SYCO_ONPOLICY" \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_OFFPOLICY" "$SYCO_ONPOLICY" \
    --results-csv ./results/transfer_olmo3-7b_layer18_with_onpolicy.csv \
    --verbose

# Output: results/transfer_olmo3-7b_layer18_with_onpolicy.csv
# On-policy CV AUROC: 0.947 (vs off-policy 0.940)
# Off-policy↔on-policy transfer: 0.959/0.969 (near-perfect mutual transfer)
# FLEED→Syco(on) transfer: 0.63-0.82 range (same pattern as FLEED→Syco(off))

# Base model (reuses existing 6 probes, only trains on-policy probe)
uv run python scripts/train_test_probes.py \
    --model-name olmo3-7b-purebase \
    --features-file ./results/activations_got_plus_fleed_olmo3-7b-purebase_layers10-25.h5 \
    --layer-idx 18 \
    --probe-type lr --regularization 0.001 \
    --train-feature-type last --test-feature-type last \
    --use-scaler true --balance-groups false \
    --train-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_OFFPOLICY" "$SYCO_ONPOLICY" \
    --test-tasks claims__definitional_gemini_600_full claims__evidential_gemini_600_full \
        claims__fictional_gemini_600_full claims__logical_gemini_600_full got__best "$SYCO_OFFPOLICY" "$SYCO_ONPOLICY" \
    --results-csv ./results/transfer_olmo3-7b-purebase_layer18_with_onpolicy.csv \
    --verbose

# Output: results/transfer_olmo3-7b-purebase_layer18_with_onpolicy.csv
# On-policy CV AUROC: 0.964 (vs off-policy 0.961)
# Off-policy↔on-policy transfer: 0.974/0.978 (near-perfect mutual transfer)
# Key finding: on-policy and off-policy sycophancy probes capture essentially the same direction
```
