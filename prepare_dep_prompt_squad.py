from datasets import load_from_disk, DatasetDict
from monty.serialization import loadfn
import lxml.html

import StanfordDependencies
sd = StanfordDependencies.get_instance()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int)
parser.add_argument("--num_chunks", type=int, default=10)
args = parser.parse_args()
print(args.index)

label_to_phrase = loadfn("./dependency_tags.json")
sep_token = "<lsep>"
do_convert = True


#def convert_tree_to_prompt(ctrees):
def convert_tree_to_prompt(dtree_conlls):
    dtree_prompts = []
    for dtree_conll in dtree_conlls:
        lines = dtree_conll.strip().split("\n")
        pt = []
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
            rel = tabs[7]
            if rel not in label_to_phrase:
                label_to_phrase[rel] = "unknown"
            dep_nl = label_to_phrase[rel]
            if target_idx not in idx_to_word:
                idx_to_word[target_idx] = "NULL"
            pt.append(f"\"{word}\" is a {dep_nl} of \"{idx_to_word[target_idx]}\"")

        dtree_prompts.append("; ".join(pt) + ".")

    return " ".join(dtree_prompts)

def process_func(e):

    for key in ("answer", "context", "question"):
        dtree_conll = e[f"{key}_dtree"]
        e[f"{key}_prompt"] = convert_tree_to_prompt(dtree_conll)

        #answer_prompts = [convert_tree_to_prompt(ac) for ac in answer_ctrees]
        #answer_prompts = convert_tree_to_prompt(answer_ctrees)
        #answer_prompts = " ".join(answer_prompts)

    #e["input"] = f"Input: {e['answer']} {e['context']} Parsed input: {e['answer_prompt']} {e['context_prompt']}"
    #e["output"] = f"Output: {e['question']} Parsed output: {e['question_prompt']}"
    e["input"] = f"Input: {e['answer']} {e['context']}"
    e["output"] = f"Parsed input: {e['answer_prompt']} {e['context_prompt']}"

    return e


d = load_from_disk(f"/scratch/aae15163zd/inputs/squad-parsed-dep-conll/merged/")
d = d.map(
    process_func,
    batched=False,
    num_proc=40
)
d.save_to_disk(f"/scratch/aae15163zd/inputs/squad-parsed-dep-prompt-I")
