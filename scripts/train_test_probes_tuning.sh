#!/bin/bash

set -e  # Exit on any error
cd ..

### TUNING PROCESS ###
# fix to lr + regularization=1 + average, find the top 5 layers
# tune token aggregation + probe type, find best layer and best probe design (33 + average + lr)
# tune the regularization weights
# fix average + lr + weights

# # EXP#1 TUNING Step 1: llama-70b-3.3
# python ./scripts/train_test_probes.py \
# --probe-type lr \
# --regularization 1 \
# --use-scale true \
# --balance-groups false \
# --layer-idx 3 8 13 18 23 28 33 38 43 48 53 58 63 68 73 78 \
# --train-feature-type average \
# --verbose \
# --model-name llama-70b-3.3 \
# --features-file ./results/extracted_feats_all_layers_llama-70b-3.3.h5 \
# --results-csv ./results/results_llama-70b-3.3_all_layers.csv

# # # EXP#2 TUNING Step 2: llama-70b-3.3
# for FEATURE in last average all; do
#     for USE_SCALE in true false; do
#         for PROBE_TYPE in diffmean lda; do
#             python ./scripts/train_test_probes.py \
#             --probe-type ${PROBE_TYPE} \
#             --regularization 0 \
#             --layer-idx 33 38 68 73 78 \
#             --train-feature-type ${FEATURE} \
#             --use-scale ${USE_SCALE} \
#             --verbose \
#             --model-name llama-70b-3.3 \
#             --features-file ./results/extracted_feats_all_layers_llama-70b-3.3.h5 \
#             --results-csv ./results/results_llama-70b-3.3_all_layers.csv
#         done

#         python ./scripts/train_test_probes.py \
#             --probe-type lr \
#             --regularization 1e-4 \
#             --layer-idx 33 38 68 73 78 \
#             --train-feature-type ${FEATURE} \
#             --use-scale ${USE_SCALE} \
#             --verbose \
#             --model-name llama-70b-3.3 \
#             --features-file ./results/extracted_feats_all_layers_llama-70b-3.3.h5 \
#             --results-csv ./results/results_llama-70b-3.3_all_layers.csv
#     done
# done

# # EXP#3 TUNING Step 3: tuning for regularization weight; llama-70b-3.3
# for REG in 1e-6 1e-4 1e-2 1 100 10000; do
#     MODEL=llama-70b-3.3
#     python ./scripts/train_test_probes.py \
#     --probe-type lr \
#     --regularization ${REG} \
#     --layer-idx 33 \
#     --train-feature-type average \
#     --use-scale true \
#     --balance-groups false \
#     --verbose \
#     --model-name ${MODEL} \
#     --features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
#     --results-csv ./results/results_${MODEL}_all_layers.csv
# done

# # EXP#4 TUNING Step 4: layer tuning for other models - llama 70b
# for MODEL in llama-70b-3.3 llama-70b-base; do
#     python ./scripts/train_test_probes.py \
#     --probe-type lr \
#     --regularization 1e-4 \
#     --layer-idx 3 8 13 18 23 28 33 38 43 48 53 58 63 68 73 78 \
#     --train-feature-type average \
#     --use-scale true \
#     --balance-groups false \
#     --verbose \
#     --model-name ${MODEL} \
#     --features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
#     --results-csv ./results/results_${MODEL}_all_layers.csv
# done

# # EXP#4 TUNING Step 4: layer tuning for other models - llama 70b
# for MODEL in llama-8b llama-8b-base; do
#     python ./scripts/train_test_probes.py \
#     --probe-type lr \
#     --regularization 1e-4 \
#     --layer-idx {0..31} \
#     --train-feature-type average \
#     --use-scale true \
#     --balance-groups false \
#     --verbose \
#     --model-name ${MODEL} \
#     --features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
#     --results-csv ./results/results_${MODEL}_all_layers.csv
# done

# # EXP#4 - TUNING Step 4: layer tuning for other models - qwen 14b
# for MODEL in qwen-14b qwen-14b-base; do
#     python ./scripts/train_test_probes.py \
#     --probe-type lr \
#     --regularization 1e-4 \
#     --layer-idx 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 \
#     --train-feature-type average \
#     --use-scale true \
#     --balance-groups false \
#     --verbose \
#     --model-name ${MODEL} \
#     --features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
#     --results-csv ./results/results_${MODEL}_all_layers.csv
# done

# # EXP#4 TUNING Step 4: layer tuning for other models - qwen 7b
# for MODEL in qwen-7b qwen-7b-base; do
#     python ./scripts/train_test_probes.py \
#     --probe-type lr \
#     --regularization 1e-4 \
#     --layer-idx {0..27} \
#     --train-feature-type average \
#     --use-scale true \
#     --balance-groups false \
#     --verbose \
#     --model-name ${MODEL} \
#     --features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
#     --results-csv ./results/results_${MODEL}_all_layers.csv
# done