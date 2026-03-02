# Commands Run

## Notes

- `data/sycophancy_v2/llama-8b/raw/raw_60q_36bios_seed42_bio_ranking_run.json` (8640 records, 60 questions × 36 bios × 4 answers) is from the bio template ranking run using all 36 bio templates on 60 sampled questions. This was the run used to produce `data/sycophancy_v2/llama-8b/bio_sycophancy_analysis.txt` via `analyze_bio_sycophancy.py`. It is NOT a production sycophancy dataset — it was used to rank bios by mean sycophancy shift to select the top-performing templates.

## Sycophancy V2 Commands

### Full generation: llama-8b with top-12 bios (2026-03-02)

```bash
uv run python scripts/create_sycophancy_dataset_v2.py generate \
    --model llama-8b \
    --base-confidence-csv data/sycophancy_v2/llama-8b-base/base_confidence/llama-8b-base.csv \
    --instruct-confidence-csv data/sycophancy_v2/llama-8b/instruct_confidence/llama-8b_0shot.csv \
    --bio-templates data/sycophancy_v2/bio_templates_llama_8b.json \
    --output-dir data/sycophancy_v2 \
    --seed 123172

# Output: data/sycophancy_v2/llama-8b/raw/raw_1350q_12bios_seed123172.json

uv run python scripts/create_sycophancy_dataset_v2.py pair \
    --model llama-8b \
    --instruct-confidence-csv data/sycophancy_v2/llama-8b/instruct_confidence/llama-8b_0shot.csv \
    --raw-path data/sycophancy_v2/llama-8b/raw/raw_1350q_12bios_seed123172.json \
    --output-dir data/sycophancy_v2 \
    --seed 123172
```
