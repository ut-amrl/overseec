#!/usr/bin/env bash

ENV_NAME="${1:-vllm}"
ROOT_DIR="${2:-$PWD}"  # optional: pass repo path as 2nd arg
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

USE_CONDA=1
if [[ "$ENV_NAME" == "no_conda" ]]; then
    USE_CONDA=0
fi

cd modules/planners/
python3 setup.py build_ext --inplace
cd ../..

tmux new-session -d -s overseec

if [[ $USE_CONDA -eq 1 ]]; then
    tmux send-keys -t overseec "source $CONDA_SH && conda activate $ENV_NAME" C-m
else
    tmux send-keys -t overseec "echo 'Skipping conda activation'" C-m
fi
tmux send-keys -t overseec "cd $ROOT_DIR/modules/llm && python3 vllm_server.py" C-m

tmux split-window -h -t overseec
if [[ $USE_CONDA -eq 1 ]]; then
    tmux send-keys -t overseec:.1 "source $CONDA_SH && conda activate $ENV_NAME" C-m
else
    tmux send-keys -t overseec:.1 "echo 'Skipping conda activation'" C-m
fi
tmux send-keys -t overseec:.1 "cd $ROOT_DIR/frontend && python3 app.py" C-m

tmux attach -t overseec
