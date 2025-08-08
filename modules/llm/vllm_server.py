import os
import argparse
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List

app = FastAPI()
llm = None

class InferenceRequest(BaseModel):
    texts: List[str]

@app.post("/generate")
async def generate_response(request: InferenceRequest):
    if not request.texts:
        return {"error": "No text prompts provided!"}
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=2048,
        top_p=1.0,
        top_k=-1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    outputs = llm.generate(request.texts, sampling_params=sampling_params)
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    num_gpus = len([d for d in args.cuda.split(",") if d != ""])

    global llm
    llm = LLM(
        model=args.model,
        tensor_parallel_size=max(1, num_gpus),
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
    )

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
