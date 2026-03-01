# Project Memory

## Probe Pipeline Architecture
- Feature extraction uses nnsight (`scripts/extract_activation.py` → `src/extract_act.py`)
- For sycophancy, detection mask selects only the **answer letter token** (single char after prefilled `(`)
- Probe trains on hidden-state activations at that token, not logits
- Pipeline: datasets.py → tokenized_data.py → extract_activation.py → train_test_probes.py

## Sycophancy V2 Pipeline
- `scripts/create_sycophancy_dataset_v2.py`: generate + pair subcommands
- `data/sycophancy_v2/bio_templates.json`: templates with `{question_subject}` and `{answer_letter}` placeholders
- Uses batched HuggingFace inference (not nnsight), `AutoModelForCausalLM`
- Pre-filters questions by base + instruct confidence CSVs
- `scripts/analyze_bio_sycophancy.py`: ranks bio templates by sycophancy shift

## Bio Template Categories (ranked by effectiveness on llama-8b)
1. stakes_urgency — time pressure / consequences
2. confidence_dominance — aggressive conviction without credentials
3. self_asserted_competence — informal expertise claims
4. credential_authority — formal academic credentials
5. emotional_vulnerability — distress / struggle
6. rhetorical_pressure / social_proof / casual_baseline / emotional_enthusiasm — weaker

## Environment
- Use `source /root/.venv/bin/activate` before running python commands (faster than local venv)
- CLAUDE.md says use `uv run` instead of `python`
