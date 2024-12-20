from datasets import load_from_disk
import json

dep_label_to_id = {}

def process_func(example):

    for key in ("context", "answer"):
        dep_labels = example[f"{key}_dep_labels"]
        for dep_label in dep_labels:
            if dep_label not in dep_label_to_id:
                dep_label_to_id[dep_label] = len(dep_label_to_id)

    return example

d = load_from_disk("./dep_labels")
d = d.map(
    process_func,
    batched=False,
    num_proc=1,
)
with open("./dep2id.json", "w") as f:
    json.dump(dep_label_to_id, f)
