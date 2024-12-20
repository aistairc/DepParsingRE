import os
from datasets import load_from_disk, DatasetDict


def process_func(example):

    #answer = example["answer"]
    #context = example["context"]
    #question = example["question"]

    #answer_dep_prompt = example["answer_prompt"]
    #context_dep_prompt = example["context_prompt"]
    #question_dep_prompt = example["question_prompt"]

    example["answer_context_ctree"] = f"{example['answer_ctree']} {example['context_ctree']}"

    #example["input"] = f"Input: {answer} {context}"
    #example["output"] = f"Parsed input: {answer_dep_prompt} {context_dep_prompt}"

    return example

home = os.path.expanduser("~")
#d = load_from_disk("/scratch/aae15163zd/inputs/squad-parsed-dep-prompt/")
d = load_from_disk(os.path.join(home, "grp/inputs/squad-parsed-con-parenthesis"))
d = d.map(
    process_func,
    batched=False,
    num_proc=40
)
print(d["train"][:3])
d.save_to_disk(os.path.join(home, "grp/ACL24/squad/parsed-con"))
