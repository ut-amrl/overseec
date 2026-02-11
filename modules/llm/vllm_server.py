import os
import argparse
import torch
import uvicorn
import sys  # <-- 1. Import sys to exit cleanly
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List
from enum import Enum

class LLMZoo(Enum):
    GEMMA_2_27B_IT = "gemma-2-27b-it"
    GEMMA_3_27B_IT = "gemma-3-27b-it"
    QWEN_2_5_CODER_14B_INSTRUCT = "Qwen2.5-Coder-14B-Instruct"

    def get_llm_name(self):
        return {
            LLMZoo.GEMMA_2_27B_IT: "google/gemma-2-27b-it",
            LLMZoo.GEMMA_3_27B_IT: "google/gemma-3-27b-it",
            LLMZoo.QWEN_2_5_CODER_14B_INSTRUCT: "Qwen/Qwen2.5-Coder-14B-Instruct",
        }[LLMZoo(self.value)]
    

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
    
    parser.add_argument("--model", type=LLMZoo, default=None, choices=list(LLMZoo), help="The model to use. If not provided, an interactive menu will be shown.")
        
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8057)
    args = parser.parse_args()

    if args.model is None:
        models = list(LLMZoo)
        print("Please choose a model to run:")
        for i, model in enumerate(models):
            print(f"  [{i + 1}] {model.value}")
        
        choice = -1
        while not (1 <= choice <= len(models)):
            try:
                raw_choice = input(f"Enter a number (1-{len(models)}): ")
                choice = int(raw_choice)
                if not (1 <= choice <= len(models)):
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        args.model = models[choice - 1]
        print(f"\n🚀 Starting server with model: {args.model.value}\n")


    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    num_gpus = len([d for d in args.cuda.split(",") if d != ""])

    global llm
    llm = LLM(
        model=args.model.get_llm_name(),
        tensor_parallel_size=max(1, num_gpus),
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
        max_model_len=8192,  # Cap context length to fit in available KV cache memory
        gpu_memory_utilization=0.90,  # Use more GPU memory for KV cache
    )

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()