docker run --gpus all -it --rm \
  -v .:/workspace/overseec \
  -v ./checkpoints:/workspace/overseec/checkpoints \
  -w /workspace/overseec \
  -p 8000:8000 -p 5000:5000 -p 5002:5002 \
  --ipc=host \
  overseec-cu128