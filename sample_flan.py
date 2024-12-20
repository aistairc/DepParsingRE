import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from collections import defaultdict
from more_itertools import chunked
import transformers
from transformers import AutoTokenizer, AutoConfig

#from huggingface_hub import login
#login("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")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=42)
args = parser.parse_args()

configs = {
    "muffin": {"dataset_name": "DataProvenanceInitiative/flan2021_submix_original",
        "task_source": "Flan2021", "proportion": 0.460, "max_cap": 30000, "num_samples_for_check": 5000},
    "t0": {"dataset_name": "DataProvenanceInitiative/t0_submix_original",
        "task_source": "P3", "proportion": 0.279, "max_cap": 20000, "num_samples_for_check": 1000},
    "cot": {"dataset_name": "DataProvenanceInitiative/cot_submix_original",
        "task_source": "CoT", "proportion": 0.018, "max_cap": 100000, "num_samples_for_check": 500},
    "niv2": {"dataset_name": "DataProvenanceInitiative/niv2_submix_original",
        "task_source": "NIv2", "proportion": 0.242, "max_cap": 5000, "num_samples_for_check": 10000},
}

output_cache_dir = f"~/scr/cache/all_flan_seed{args.seed}"
if not os.path.exists(output_cache_dir):
    os.makedirs(output_cache_dir)

all_capped_samples = {}
counter = defaultdict(lambda: 0)
for k, v in configs.items():

    d = load_from_disk(os.path.join(output_cache_dir, k))
    #d = d.select(range(v["num_samples_for_check"]))

    capped_samples = []
    for e in tqdm(d, desc=f"Capping {k} out of {len(d)}"):
        # Chack maximum cap
        if counter[e["task_name"]] >= v["max_cap"]:
            continue
        capped_samples.append(e)
        counter[e["task_name"]] += 1

    all_capped_samples[k] = capped_samples

weights = [len(v) / configs[k]["proportion"] for k, v in all_capped_samples.items()]
min_weight = min(weights)
print(min_weight)

weighted_samples = []
for k, v in all_capped_samples.items():
    num_samples = int(min_weight * configs[k]["proportion"])
    print(k, num_samples)
    weighted_samples += v[:num_samples]

weighted_d = Dataset.from_list(weighted_samples)
weighted_d.save_to_disk(os.path.join(output_cache_dir, "_weighted"))
