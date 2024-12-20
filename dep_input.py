from datasets import load_from_disk
from transformers import AutoTokenizer
import json
import numpy as np

t = AutoTokenizer.from_pretrained("google/t5-large-lm-adapt")
max_length = 512
target_max_length = 128
with open("./dep2id.json") as f:
    dep_label_to_id = json.load(f)
null_id = len(dep_label_to_id)

def process_func(example):

    answer_context_dep_conlls = example["answer_dtree"] + example["context_dtree"]
    #dep_conlls = example[f"{key}_dtree"]
    offset_d = {}
    text = ""
    for dep_conll in answer_context_dep_conlls:
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
                target_word = "NULL"
            else:
                target_word = idx_to_word[target_idx]
            rel = tabs[7]
            rel_id = dep_label_to_id[rel]

            for i in range(len(text), len(text)+len(word)+1):
                offset_d[i] = {
                    "dep_label": rel,
                    "dep_label_id": rel_id,
                    "target_word": target_word,
                    "target_word_id": t.encode(target_word, add_special_tokens=False)[0] # Only use first token
                }
            text += word + " "

    output = t(text, padding="max_length", truncation=True, max_length=max_length, return_offsets_mapping=True)
    dep_label_ids = []
    target_word_ids = []
    for ii, off in zip(output.input_ids, output.offset_mapping):
        if off[0] == off[1]: # Special tokens
            dep_label_ids.append(null_id)
            target_word_ids.append(ii)
        else:
            dep_label_ids.append(offset_d[off[0]]["dep_label_id"])
            target_word_ids.append(offset_d[off[0]]["target_word_id"])
    dep_label_ids = np.array(dep_label_ids)
    target_word_ids = np.array(target_word_ids)
    stacked_inputs = np.stack([output.input_ids, dep_label_ids, target_word_ids], 0)
    #example["input_ids"] = output.input_ids
    example["input_ids"] = stacked_inputs
    example["attention_mask"] = output.attention_mask
    #example["dep_label_ids"] = dep_label_ids
    #example["target_word_ids"] = target_word_ids

    question_output = t(text_target=example["question"], padding="max_length", truncation=True, max_length=target_max_length)
    example["labels"] = question_output.input_ids
    example["labels"] = [(l if l != t.pad_token_id else -100) for l in example["labels"]]

    return example


d = load_from_disk("/groups/gac50543/migrated_from_SFA_GPFS/asada/inputs/squad-parsed-dep-conll/merged")
columns = d["train"].column_names
d = d.map(
    process_func,
    batched=False,
    num_proc=40,
    remove_columns=columns,
)
d.save_to_disk("./bar")
