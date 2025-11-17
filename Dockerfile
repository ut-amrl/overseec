FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HUGGINGFACE_HUB_CACHE=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    TORCH_HOME=/opt/torch \
    PYTHONPATH=/workspace:$PYTHONPATH \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv python3-gdal\
    git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 curl ca-certificates wget tmux && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install torchvision --index-url https://download.pytorch.org/whl/cu128

RUN TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 pip install FastGeodis --no-cache-dir --no-build-isolation

RUN python - <<'PY'
from transformers import AutoModelForSemanticSegmentation, AutoProcessor, AutoModel
import torch, os
AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
AutoModel.from_pretrained("CIDAS/clipseg-rd64-refined")
AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
torch.hub.set_dir(os.environ.get("TORCH_HOME", "/opt/torch"))
torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
PY

RUN mkdir -p checkpoints && \
    cd checkpoints && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

WORKDIR /workspace
CMD ["bash"]
