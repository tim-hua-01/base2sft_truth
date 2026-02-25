import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import DialogueDataset
from src.tokenized_data import TokenizedDataset
from src.models import get_model_and_tokenizer
from src import utils

import h5py
import numpy as np
import pandas as pd
import csv

MODEL_NAME = 'llama-8b'
ALL_TASKS = ['repe_honesty__plain',
    'claims__definitional_gemini_600_full',
    'claims__evidential_gemini_600_full',
    'claims__fictional_gemini_600_full',
    'claims__logical_gemini_600_full',
    'internal_state__animals',
    'internal_state__cities',
    'internal_state__companies',
    'internal_state__elements',
    'internal_state__facts',
    'internal_state__inventions',
    'got__best',
    'sycophancy__mmlu_stem_conf_part_1',
    'sycophancy__mmlu_stem_conf_part_2',
    'sycophancy__mmlu_stem_conf_all',
    'ethics__commonsense',]

### open tokenizer
_, tokenizer = get_model_and_tokenizer(MODEL_NAME, "../deception_detection/data/huggingface/", omit_model = True)

### open feats
extracted_feats = h5py.File(f"./results/extracted_feats_all_layers_{MODEL_NAME}.h5", 'r')

### open all datasets
all_datasets = {}
for dataset_name in ALL_TASKS:
    print(dataset_name)
    all_datasets[dataset_name] = DialogueDataset(dataset_name, MODEL_NAME)


### get domain labels
domain_labels = {'claims__evidential_gemini_600_full': [],
                'claims__logical_gemini_600_full': [],
                'got__best': []}

# get domain labels for evidential claims
csv_file = pd.read_csv(f"./data/claims/claims__evidential_gemini_600_full.csv")
for domain_label in csv_file['Domain']:
    # append twice for true/false
    domain_labels['claims__evidential_gemini_600_full'].append(domain_label)
    domain_labels['claims__evidential_gemini_600_full'].append(domain_label)

# get domain labels for logical claims
csv_file = pd.read_csv(f"./data/claims/claims__logical_gemini_600_full.csv")
for domain_label in csv_file['Domain']:
    # append twice for true/false
    domain_labels['claims__logical_gemini_600_full'].append(domain_label)
    domain_labels['claims__logical_gemini_600_full'].append(domain_label)

# get domain labels for got__best
csv_file_paths = ["./data/geometry_of_truth/cities.csv",
                    "./data/geometry_of_truth/neg_cities.csv",
                    "./data/geometry_of_truth/larger_than.csv",
                    "./data/geometry_of_truth/smaller_than.csv"]
for _path in csv_file_paths:
    _domain = _path.split('/')[-1].split('.')[0]
    with open(_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        domain_labels['got__best'].extend([_domain for _ in range(len(list(reader)))])

output_file = f'./results/activations_layer_15_{MODEL_NAME}.h5'
# output_file = f'./results/activations_all_layers_{MODEL_NAME}.h5'
with h5py.File(output_file, 'a') as f:
    # for layer in extracted_feats.keys():
    for layer in ['layer_15']:
        print(f'Processing layer: {layer}')
        # Use require_group instead of create_group to avoid errors if it already exists
        layer_group = f.require_group(layer)
        
        for dataset_name in ALL_TASKS:
            if 'part' in dataset_name: # skip part datasets 
                continue 
            
            # Create a subgroup for each task
            assert dataset_name not in layer_group, f"{dataset_name} already exists in layer {layer}!"
                
            print(dataset_name)
            prep_data = utils.prepare_data(dataset_name, layer, extracted_feats, 
                                        all_datasets, tokenizer, feature_type='average')
            X, y = prep_data.X, prep_data.y
            
            task_group = layer_group.create_group(dataset_name)
            task_group.create_dataset('X', data=X, compression="gzip")
            task_group.create_dataset('y', data=y, compression="gzip")

            # add domain labels if applicable
            if dataset_name in domain_labels:
                domain_label_data = domain_labels[dataset_name]
                # Convert to numpy array of fixed-length strings
                domain_label_array = np.array(domain_label_data, dtype='S')
                task_group.create_dataset('domain_labels', data=domain_label_array, compression="gzip")

