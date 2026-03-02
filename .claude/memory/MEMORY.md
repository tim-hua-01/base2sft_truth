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
- `scripts/analyze_bio_sycophancy.py`: ranks bio templates by sycophancy shift
- Latest llama-8b run: 470 pairs from 1350 questions, see `commands_run.md` for exact commands
- batch_size=64 works on L40S (46GB); 128+ OOMs due to variable sequence lengths across batches

## Bio Template Categories (ranked by effectiveness on llama-8b)
1. stakes_urgency — time pressure / consequences
2. confidence_dominance — aggressive conviction without credentials
3. self_asserted_competence — informal expertise claims
4. credential_authority — formal academic credentials
5. emotional_vulnerability — distress / struggle
6. rhetorical_pressure / social_proof / casual_baseline / emotional_enthusiasm — weaker

## Key Files
- `commands_run.md`: tracks all dataset generation commands with provenance notes
- `next_steps.md`: plan to adapt probing pipeline for v2 CSV format + GroupKFold

## Environment
- Use `uv run` instead of `python` (project uses uv for dependency management)
- GPU: L40S 46GB; batch_size=64 safe for 8B model, 128+ risky
