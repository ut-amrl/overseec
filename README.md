# overseec

## Conda Installation
```bash
conda create -n <env_name> python=3.10 -y

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

pip install FastGeodis --no-build-isolation

pip install -r requirements.txt

mkdir checkpoints
cd checkpoints
curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L
curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (smallest)
curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

chmod +x start_overseec.sh

./start_overseec.sh <env_name>
```

## Docker

```bash
docker build -t overseec-cu128 .

docker run --gpus all -it --rm \
  -v .:/workspace/overseec \
  -v ./checkpoints:/workspace/overseec/checkpoints \
  -w /workspace/overseec \
  -p 8000:8000 -p 5000:5000 -p 5002:5002 \
  --ipc=host \
  overseec-cu128

./start_overseec.sh no_conda # inside docker
```
