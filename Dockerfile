FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HUGGINGFACE_HUB_CACHE=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    TORCH_HOME=/opt/torch \
    PYTHONPATH=/workspace \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    libgdal-dev gdal-bin libproj-dev libgeos-dev \
    git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 curl ca-certificates wget tmux && \
    apt-get purge -y python3-gdal && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install "numpy>=2.0.0" "cython"

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install --no-build-isolation GDAL==$(gdal-config --version) --no-binary GDAL

RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install torchvision --index-url https://download.pytorch.org/whl/cu128

RUN TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 pip install FastGeodis --no-cache-dir --no-build-isolation

RUN python - <<'PY'
from transformers import AutoModelForSemanticSegmentation, AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch, os
AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
AutoModel.from_pretrained("CIDAS/clipseg-rd64-refined")
AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
torch.hub.set_dir(os.environ.get("TORCH_HOME", "/opt/torch"))
torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
PY
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

WORKDIR /workspace
CMD ["bash"] 
