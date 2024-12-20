from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

prefix = "Perform dependency parsing on the following text. "

model_path = "/scratch/aae15163zd/outputs/instruction-tuning/dep-con-0-10-t5-base-lma"
t = AutoTokenizer.from_pretrained(model_path)
m = AutoModelForSeq2SeqLM.from_pretrained(model_path)

#cache_dir = "/scratch/aae15163zd/cache/parsing/wikipedia-10-20-dep/"
#d = load_from_disk(cache_dir)
#d = d.select(range(100))
#d.save_to_disk("wikipedia-10-20-dep-for-check")
d = load_from_disk("wikipedia-10-20-dep-for-check")

for e in d:
    print(e["sentence"])
    model_input = t(prefix + e["sentence"], return_tensors="pt")
    model_output = m.generate(model_input.input_ids)
    pred_tokens = t.decode(model_output.logit.argmax(-1), skip_special_tokens=True)

    print(pred_tokens)
    print(e["dtree"])

