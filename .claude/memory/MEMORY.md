# Project Memory

## Probe Pipeline Architecture
- Feature extraction uses nnsight (`scripts/extract_activation.py` → `src/extract_act.py`)
- For sycophancy, detection mask selects only the **answer letter token** (single char after prefilled `(`)
- Probe trains on hidden-state activations at that token, not logits
- Pipeline: datasets.py → tokenized_data.py → extract_activation.py → train_test_probes.py

## Sycophancy V2 Pipeline
- `scripts/create_sycophancy_dataset_v2.py`: generate + pair subcommands
- `data/sycophancy_v2/bio_templates.json`: full 36 templates with `{question_subject}` and `{answer_letter}` placeholders
- `data/sycophancy_v2/bio_templates_llama_8b.json`: top 12 by mean_shift
- Uses batched HuggingFace inference (not nnsight), `AutoModelForCausalLM`
- Pre-filters questions by base + instruct confidence CSVs
- `--max-bios-per-question`: randomly samples N bios per question from the full pool
- `--max-pairs-per-question`: in pair stage, samples up to N valid pairs per question (default: all)
- `scripts/analyze_bio_sycophancy.py`: ranks bio templates by sycophancy shift
- Latest llama-8b run: 834 pairs from 470 questions (max 2/q), see `commands_run.md`
- batch_size=64 works on L40S/A40 (46GB); 128+ OOMs due to variable sequence lengths

## Sycophancy V2 Probing (implemented 2026-03-02)
- `src/datasets.py`: `sycophancy_v2` loader reads pairs CSV, reconstructs dialogues from MMLU + bio templates, returns `(dialogues, labels, group_ids)` 3-tuple
- Task name format: `sycophancy_v2__data/sycophancy_v2/llama-8b/pairs/<csv_filename>.csv`
- `src/utils.py`: `PreparedData.group_ids` field, propagated through `prepare_sample_data`/`prepare_data`
- `src/probe_trainer.py`: `GroupKFold` used when `group_ids` present (prevents data leakage from paired samples)
- Current dataset: 1668 dialogues (834 pairs × 2), 834 groups
- CSV: `data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv`

## Bio Template Categories (ranked by effectiveness on llama-8b)
1. stakes_urgency — time pressure / consequences
2. confidence_dominance — aggressive conviction without credentials
3. self_asserted_competence — informal expertise claims
4. credential_authority — formal academic credentials
5. emotional_vulnerability — distress / struggle
6. rhetorical_pressure / social_proof / casual_baseline / emotional_enthusiasm — weaker

## Key Files
- `commands_run.md`: tracks all dataset generation commands with provenance notes
- `next_steps.md`: current status and next steps

## Environment
- Use `uv run` instead of `python` (project uses uv for dependency management)
- GPU: A40 46GB (or L40S 46GB); batch_size=64 safe for 8B model
- nnsight extraction uses batch_size=1 for sycophancy tasks (variable length prompts)
