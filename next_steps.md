# Next Steps: Adapt probing pipeline for sycophancy_v2

## Context
The sycophancy_v2 pipeline now produces paired CSV data (`data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_05:34:21.csv`, 470 pairs). The probing pipeline needs a loader for this format, and CV splitting must keep pairs from the same question in the same fold to prevent data leakage.

## Changes needed

### 1. Add `sycophancy_v2` loader in `src/datasets.py` (~line 514)

New `elif base == 'sycophancy_v2'` block in `load_from_prewritten()`.

- **Variant**: full relative path to the pairs CSV (e.g. `data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_05:34:21.csv`)
- **Bio templates**: hardcoded default `data/sycophancy_v2/bio_templates.json` (full 36, CSV bio_id will match)
- **MMLU questions**: loaded from HuggingFace `cais/mmlu` by parsing `question_id` (format: `{task}_{idx}`)
- **Per CSV row** produces 2 dialogues (same Message structure as old sycophancy loader):
  - **Sycophantic** (DECEPTIVE): user=bio(syco_answer)+question+choices, assistant prefix + syco answer letter
  - **Honest** (HONEST): user=bio(correct_answer)+question+choices, assistant prefix + correct answer letter
- **Returns** `(dialogues, labels, group_ids)` — group_ids = `[0, 0, 1, 1, 2, 2, ...]` so both dialogues from a question share a group

User content reconstruction matches `create_bio_chat_prompt` logic from `create_sycophancy_dataset_v2.py:132-139`:
```
bio_text + "\n\nHere's the question:\n\n" + question + "\n\nChoices:\n(A) ...\n(B) ...\n\nPlease provide your answer."
```

Dialogue structure (matching old format):
```
Message("user", user_content, False)
Message("assistant", "I believe the best answer is (", False)
Message("assistant", letter, True)   # single char, detect=True
```

### 2. Add optional `group_ids` to `DialogueDataset` and `PreparedData`

**`DialogueDataset.__init__`** (`src/datasets.py:48`):
- For sycophancy_v2: `load_from_prewritten` returns 3-tuple `(dialogues, labels, group_ids)`
- For all others: returns 2-tuple as before, `self.group_ids = None`

**`PreparedData`** (`src/utils.py:195`):
- Add `group_ids: Optional[np.ndarray] = None`

**`prepare_sample_data`** (`src/utils.py:218`):
- Accept optional `group_ids`, filter/propagate alongside y and dataset_ids

**`prepare_data`** (`src/utils.py:331`):
- Read `group_ids` from dataset, pass through to PreparedData

### 3. Use `GroupKFold` when group_ids present

**`ProbeTrainer._get_cv_splits`** (`src/probe_trainer.py:179`):
- Accept optional `group_ids`
- If present: `GroupKFold(n_splits=self.n_folds).split(X, y, groups=group_ids)`
- Else: existing `KFold` behavior

**`train_probe_single_layer`** (`src/probe_trainer.py:116`):
- Pass `data.group_ids` to `_get_cv_splits`

## Files to modify
1. `src/datasets.py` — new sycophancy_v2 loader, group_ids on DialogueDataset
2. `src/utils.py` — group_ids on PreparedData, propagation through prepare_sample_data/prepare_data
3. `src/probe_trainer.py` — GroupKFold import + conditional use

## Verification
```python
ds = DialogueDataset('sycophancy_v2__data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_05:34:21.csv', 'llama-8b')
assert len(ds) == 940  # 470 pairs × 2
assert ds.group_ids is not None
assert ds.group_ids[0] == ds.group_ids[1]  # pair shares group
assert ds.group_ids[2] == ds.group_ids[3]
assert ds.group_ids[0] != ds.group_ids[2]  # different questions differ
```
