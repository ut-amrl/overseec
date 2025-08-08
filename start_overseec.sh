#!/usr/bin/env bash

ENV_NAME="${1:-vllm}"
ROOT_DIR="${2:-$PWD}"  # optional: pass repo path as 2nd arg
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

cd modules/planners/
python3 setup.py build_ext --inplace
cd ../..

tmux new-session -d -s overseec
tmux send-keys -t overseec "source $CONDA_SH && conda activate $ENV_NAME" C-m
tmux send-keys -t overseec "cd $ROOT_DIR/modules/llm && python3 vllm_server.py" C-m

tmux split-window -h -t overseec
tmux send-keys -t overseec:.1 "source $CONDA_SH && conda activate $ENV_NAME" C-m
tmux send-keys -t overseec:.1 "cd $ROOT_DIR/frontend && python3 app.py" C-m
tmux attach -t overseec
