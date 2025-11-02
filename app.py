from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "the-verdict-finetuned"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_Length: int = 100

@app.post("/generate")
def generate_text(req: GenerateRequest):
    input = tokenizer(req.prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **input,
            max_length=req.max_Length + input.input_ids.shape[1],
            do_sample=True,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,  #confidence of picking the next wor
            top_p=req.top_p,  #nucleus sampling, cumulative probability of 0.95
            top_k=req.top_k, #keeps the 50 most likely next words
            eos_token_id=tokenizer.eos_token_id, #stop at end of sentence token
            pad_token_id=tokenizer.pad_token_id #to avoid warning since no pad token in gpt2
        )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"output": output}
