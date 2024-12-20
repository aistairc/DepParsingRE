import benepar, spacy
from monty.serialization import loadfn

#sentence = "The Reichskammergericht on the other hand was often torn by matters related to confessional alliance."
sentence = "The two chanceries became combined in 1502."
print(sentence)
label_to_phrase = loadfn("./dependency_tags.json")

nlp = spacy.load('en_core_web_md')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

doc = nlp(sentence)
sent = list(doc.sents)[0]
const_out = sent._.parse_string
print(const_out)

import StanfordDependencies
sd = StanfordDependencies.get_instance(backend='subprocess')
dep_out = sd.convert_tree(const_out)
print(dep_out.as_conll())

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
            #pt.append(f"\"{word}\" is a {dep_nl} of \"{idx_to_word[target_idx]}\"")
            pt.append(f"``{word}'' is a {dep_nl} of ``{idx_to_word[target_idx]}''")

        dtree_prompts.append("; ".join(pt) + ".")
    return " ".join(dtree_prompts)

dep_prompt = convert_tree_to_prompt([dep_out.as_conll()])
print(dep_prompt)
