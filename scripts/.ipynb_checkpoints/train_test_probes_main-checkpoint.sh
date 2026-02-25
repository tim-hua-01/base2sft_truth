#!/bin/bash

set -e  # Exit on any error
cd ..

# main - llama-70b-3.3 & llama-8b
MODEL=llama-70b-3.3
LAYER=33
python ./scripts/train_test_probes.py \
--probe-type lr \
--regularization 1e-4 \
--use-scale true \
--balance-groups false \
--layer-idx ${LAYER} \
--train-feature-type average \
--verbose \
--model-name ${MODEL} \
--features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
--results-csv ./results/results_${MODEL}_all_layers.csv

# main - llama-70b-3.3 & llama-8b
MODEL=llama-8b
LAYER=13
python ./scripts/train_test_probes.py \
--probe-type lr \
--regularization 1e-4 \
--use-scale true \
--balance-groups false \
--layer-idx ${LAYER} \
--train-feature-type average \
--verbose \
--model-name ${MODEL} \
--features-file ./results/extracted_feats_all_layers_${MODEL}.h5 \
--results-csv ./results/results_${MODEL}_all_layers.csv