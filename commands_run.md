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
