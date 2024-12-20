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
def convert_tree_to_prompt(ctree):
    #assert isinstance(ctree, str)

    #print(ctree)
    #dtrees = sd.convert_trees(ctrees)
    #dtree_conlls = [dtree.as_conll() for dtree in dtrees]
    #return dtree_conlls
    try:
        dtree = sd.convert_tree(ctree)
        dtree_conll = dtree.as_conll()
    except:
        dtree_conll = ""
    return dtree_conll

    dtree_prompts = []
    for dtree_conll in dtree_conlls:
        lines = dtree_conll.split("\n")
        pt = []
        idx_to_word = {"0": "root"}
        for line in lines:
            tabs = line.split("\t")
            idx = tabs[0]
            word = tabs[1]
            idx_to_word[idx] = word

        for line in lines:
            tabs = line.split("\t")
            idx = tabs[0]
            word = tabs[1]
            target_idx = tabs[6]
            rel = tabs[7]
            if rel not in label_to_phrase:
                label_to_phrase[rel] = "unknown"
            dep_nl = label_to_phrase[rel]
            pt.append(f"\"{word}\" is a {dep_nl} of \"{idx_to_word[target_idx]}\"")

        dtree_prompts.append("; ".join(pt) + ".")

    return dtree_prompts

def process_func(example):

    for key in ("answer", "context", "question"):
        text = example[key]
        ctrees = example[f"{key}_ctree"].split(sep_token)
        dtrees = [convert_tree_to_prompt(ctree) for ctree in ctrees]
        example[f"{key}_dtree"] = dtrees
        #example[f"{key}_dtree"] = convert_tree_to_prompt(dtrees)

        #answer_prompts = [convert_tree_to_prompt(ac) for ac in answer_ctrees]
        #answer_prompts = convert_tree_to_prompt(answer_ctrees)
        #answer_prompts = " ".join(answer_prompts)

    return example


d_dict = DatasetDict()
for split in ("train", "validation", "test"):
    d = load_from_disk(f"/scratch/aae15163zd/inputs/squad-all-parsed-{split}/chunked-{args.index}-{args.num_chunks}")
    #d = load_from_disk(f"/scratch/aae15163zd/inputs/squad-all-parsed-{split}/")
    d = d.map(
        process_func,
        batched=False,
        num_proc=40
    )
    d_dict[split] = d
d_dict.save_to_disk(f"/scratch/aae15163zd/inputs/squad-parsed-dep-conll/chunked-{args.index}-{args.num_chunks}")
