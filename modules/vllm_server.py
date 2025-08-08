import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List
import os

# Set GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

app = FastAPI()

model_name = "google/gemma-2-27b-it"
# model_name = "google/codegemma-7b-it"
# model_name = "allenai/Molmo-7B-D-0924"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 

# Define Request Schema
class InferenceRequest(BaseModel):
    texts: List[str]  # List of text prompts

@app.post("/generate")
async def generate_response(request: InferenceRequest):
    if not request.texts:
        return {"error": "No text prompts provided!"}
    
    sampling_params = SamplingParams(
        temperature=0.2,                 # Lower temperature = more deterministic
        max_tokens=2048,                 # Adjust based on dictionary length
        top_p=1.0,
        top_k=-1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Just pass raw list of prompts
    outputs = llm.generate(request.texts, sampling_params=sampling_params)

    return outputs

if __name__ == "__main__":
    llm = LLM(
        model=model_name,
        tensor_parallel_size=NUM_GPUS,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
