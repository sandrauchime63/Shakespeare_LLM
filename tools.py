from transformers import AutoTokenizer
from datasets import Dataset


with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text=[line.strip() for line in f.readlines() if line.strip()]
    raw_text="\n".join(raw_text)
#print(raw_text[:500])

tokenizer=AutoTokenizer.from_pretrained("distilgpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token':'<PAD>'})


tokens=tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
#print(tokenizer[:50])

#cutting into smaller pieces for training in finetuning batch size
chunks=[]
for i in range(0, len(tokens), 128):
    chunk=tokens[i:i+128]
    if len(chunk)<128:
        chunk+=[tokenizer.pad_token_id] * (128 - len(chunk)) #padded to length 128 incase of incomplete last chunk
    chunks.append({"input_ids": chunk, "attention_mask": [1]*len(chunk)}) #attention mask of 1s since no padding within chunks

dataset=Dataset.from_list(chunks)
dataset.save_to_disk("the-verdict-dataset")