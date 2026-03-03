# Next Steps

## Completed: Adapt probing pipeline for sycophancy_v2

All three changes from the original plan are implemented and verified:

1. `src/datasets.py` — `sycophancy_v2` loader + `group_ids` on `DialogueDataset`
2. `src/utils.py` — `group_ids` on `PreparedData`, propagated through `prepare_sample_data`/`prepare_data`
3. `src/probe_trainer.py` — `GroupKFold` used when `group_ids` present

Dataset: 834 pairs (2 per question max) = 1668 dialogues, 834 groups.
CSV: `data/sycophancy_v2/llama-8b/pairs/sycophancy_pairs_26-03-02_22:30:03.csv`

## In progress: Extract activations and train probes

- Extracting activations for sycophancy_v2 on llama-8b (all 32 layers)
- Output: `results/extracted_feats_all_layers_llama-8b_sycophancy_v2.h5`
- Then: train probes with `train_test_probes.py`
