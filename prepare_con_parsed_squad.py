from datasets import load_from_disk, DatasetDict
from monty.serialization import loadfn
import lxml.html

label_to_phrase = loadfn("./constituency_tags.json")
sep_token = "<lsep>"
do_convert = False

def get_depth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d

def get_all_children(node):
    str_c = None
    space = node.text.strip().split()
    if len(space) >= 2:
        str_c = " ".join(space[1:])
    children = list(node.getchildren())
    return str_c, children

def convert_tree_to_prompt(ctree):
    assert isinstance(ctree, str)

    if not do_convert:
        return ctree

    ctree = ctree.replace("(", "<node>")
    ctree = ctree.replace(")", "</node>")
    ctree = "<data>" + ctree + "</data>"
    root = lxml.html.fromstring(ctree)

    s = ""
    for x in root.iter():
        d = get_depth(x)
        if d <= 3: continue

        x_text = x.text.strip()
        x_space = x_text.split()
        if x_space[0] not in label_to_phrase:
            label_to_phrase[x_space[0]] = "unknown"
            print(x_space[0])
            return ""

        children = list(x.getchildren())
        if len(children) == 0: continue

        #s += f"{d} "
        if d == 4:
            s += f'The {label_to_phrase[x_space[0]]} has '
        elif d == 5:
            s += f'; The {label_to_phrase[x_space[0]]} has '
        elif d > 5:
            s += f', which has '
        if len(x_space) >= 2:
            s += f'the "{" ".join(x_space[1:])}" and '

        for i, c in enumerate(children):
            #if c.text is None:
            #    batch["is_valid"] = False
            #    batch["ctree_prompt"] = ""
            #    return batch
            c_text = c.text.strip()
            c_space = c_text.split()
            #if len(c_space) == 0:
            #    batch["is_valid"] = False
            #    batch["ctree_prompt"] = ""
            #    return batch
            if c_space[0] not in label_to_phrase:
                label_to_phrase[c_space[0]] = "unknown"
            #    batch["is_valid"] = False
            #    batch["ctree_prompt"] = ""
            #    return batch

            str_c, gchildren = get_all_children(c)
            if len(gchildren) == 0 and str_c is not None:
                s += f'the {label_to_phrase[c_space[0]]} "{str_c}" '
            else:
                s += f'a {label_to_phrase[c_space[0]]} '
            s += "and "
        s = s.rstrip("and ")
    s += "."
    #batch["is_valid"] = True
    #batch["ctree_prompt"] = s + "."

    return s

def process_func(example):

    answer = example["answer"]
    answer_ctrees = example["answer_ctree"].split(sep_token)
    #print(answer_ctree)
    answer_prompts = [convert_tree_to_prompt(ac) for ac in answer_ctrees]
    answer_prompts = " ".join(answer_prompts)

    context = example["context"]
    context_ctrees = example["context_ctree"].split(sep_token)
    context_prompts = [convert_tree_to_prompt(cc) for cc in context_ctrees]
    context_prompts = " ".join(context_prompts)


    question = example["question"]
    question_ctrees = example["question_ctree"].split(sep_token)
    question_prompts = [convert_tree_to_prompt(qc) for qc in question_ctrees]
    question_prompts = " ".join(question_prompts)
    #print(example["context_ctree"].split(sep_token))

    #example["input"] = f"Input: {answer} {context}"
    #example["input"] = f"Parsed input: {answer_prompts} {context_prompts}"
    example["input"] = f"Input: {answer} {context} Parsed input: {answer_prompts} {context_prompts}"
    #example["output"] = f"Parsed input: {answer_prompts} {context_prompts}"
    example["output"] = f"Output: {question} Parsed output: {question_prompts}"
    #example["input"] = f"Input: {answer} {context}"
    #example["output"] = f"Output: {question}"
    #example["input"] = f"{answer} {context}"
    #example["output"] = f"{question}"

    return example


d_dict = DatasetDict()
for split in ("train", "validation", "test"):
    d = load_from_disk(f"/scratch/aae15163zd/inputs/squad-all-parsed-{split}/")
    d = d.map(
        process_func,
        batched=False,
        num_proc=40
    )
    d_dict[split] = d
d_dict.save_to_disk(f"/scratch/aae15163zd/inputs/squad-parsed-con-paren-Irp-Orp")
