FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HUGGINGFACE_HUB_CACHE=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    TORCH_HOME=/opt/torch \
    PYTHONPATH=/workspace:$PYTHONPATH   # add /workspace to Python path

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install --no-build-isolation FastGeodis

RUN python - <<'PY'
from transformers import AutoModelForSemanticSegmentation, AutoProcessor, AutoModel
import torch, os
AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
AutoModel.from_pretrained("CIDAS/clipseg-rd64-refined")
AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
torch.hub.set_dir(os.environ.get("TORCH_HOME", "/opt/torch"))
torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
PY

WORKDIR /workspace
CMD ["bash"]
