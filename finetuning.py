from datasets import  Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import accelerate

dataset=load_from_disk("the-verdict-dataset")
#print(type(dataset))
data_split = dataset.train_test_split(test_size=0.05) #type: ignore
train=data_split["train"]
test=data_split["test"]
#print(datasets[0])

tokenizer=AutoTokenizer.from_pretrained("distilgpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token':'PAD'})

#word_embeddings
model=AutoModelForCausalLM.from_pretrained("distilgpt2")
# If we added tokens (pad), resize embeddings
model.resize_token_embeddings(len(tokenizer))

#formats data(pad,batch,prepares labes) for training
data_collator= DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

output_dir="the-verdict-finetuned"
epoch=3
batch_size=8

training_arg=TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epoch,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-4,
    logging_steps=50, # type: ignore
    save_total_limit=2,
    fp16 = False
)

trainer=Trainer(
    model=model,
    args=training_arg,
    train_dataset=train, #type:ignore
    eval_dataset=test,   #type:ignore
    data_collator=data_collator
)

trainer.train()
trainer.save_model(output_dir)
result=trainer.evaluate()
tokenizer.save_pretrained(output_dir)
