from datasets import load_from_disk
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("google/t5-large-lm-adapt")
max_length = 512

def process_func(example):

    for key in ("context", "answer"):
        dep_conlls = example[f"{key}_dtree"]
        rels = []
        for dep_conll in dep_conlls:
            lines = dep_conll.strip().split("\n")
            idx_to_word = {"0": "root"}
            for line in lines:
                tabs = line.split("\t")
                if len(tabs) < 2: continue
                idx = tabs[0]
                word = tabs[1]
                idx_to_word[idx] = word

            for line in lines:
                tabs = line.split("\t")
                if len(tabs) < 2: continue
                idx = tabs[0]
                word = tabs[1]
                target_idx = tabs[6]
                if target_idx not in idx_to_word:
                    idx_to_word[target_idx] = "NULL"
                rel = tabs[7]
                rels.append(rel)
        example[f"{key}_dep_labels"] = rels

    #question_dep_conll = example["question_dtree"]

    return example

d = load_from_disk("/groups/gac50543/migrated_from_SFA_GPFS/asada/inputs/squad-parsed-dep-conll/merged")
d = d.map(
    process_func,
    batched=False,
    num_proc=40,
)
d.save_to_disk("./dep_labels")
