#!/usr/bin/env bash

# --- 1. Set default values ---
ENV_NAME="vllm"
ROOT_DIR="$PWD"
LLM_MODEL_ARG=""
VLLM_DEVICE_ARG="" # <-- Added new variable for the device flag

# --- 2. Parse all arguments as keyword arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-model)
      LLM_MODEL_ARG="--model $2"
      shift 2 # Consume the flag and its value
      ;;
    --env)
      ENV_NAME="$2"
      shift 2 # Consume the flag and its value
      ;;
    --dir)
      ROOT_DIR="$2"
      shift 2 # Consume the flag and its value
      ;;
    --vllm-device) # <-- Added case for the new device flag
      VLLM_DEVICE_ARG="--cuda \"$2\""
      shift 2 # Consume the flag and its value
      ;;
    *)
      # Handle unknown arguments
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# --- Your original setup and tmux logic ---
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

USE_CONDA=1
if [[ "$ENV_NAME" == "no_conda" ]]; then
    USE_CONDA=0
fi

# Make sure the root directory exists before trying to cd into subdirs
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory '$ROOT_DIR' not found."
    exit 1
fi

echo "🚀 Starting setup..."
echo "Conda Env: $ENV_NAME"
echo "Root Dir:  $ROOT_DIR"
echo "LLM Args:  ${LLM_MODEL_ARG:-'(interactive)'}"
echo "Device Args: ${VLLM_DEVICE_ARG:-'(default)'}" # <-- Added echo for new flag
echo "----------------------------------------"


cd "$ROOT_DIR/modules/planners/"
python3 setup.py build_ext --inplace
cd "$ROOT_DIR"

# --- Tmux Session ---
# Kill any existing session with the same name
tmux kill-session -t overseec 2>/dev/null || true
tmux new-session -d -s overseec

# Pane 1: VLLM Server
if [[ $USE_CONDA -eq 1 ]]; then
    tmux send-keys -t overseec "source $CONDA_SH && conda activate $ENV_NAME" C-m
else
    tmux send-keys -t overseec "echo 'Skipping conda activation'" C-m
fi

# <-- Appended the new device argument to the command
tmux send-keys -t overseec "cd $ROOT_DIR/modules/llm && VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 python3 vllm_server.py $LLM_MODEL_ARG $VLLM_DEVICE_ARG" C-m
sleep 1

# Pane 2: Frontend App
tmux split-window -h -t overseec
if [[ $USE_CONDA -eq 1 ]]; then
    tmux send-keys -t overseec:.1 "source $CONDA_SH && conda activate $ENV_NAME" C-m
else
    tmux send-keys -t overseec:.1 "echo 'Skipping conda activation'" C-m
fi
tmux send-keys -t overseec:.1 "cd $ROOT_DIR/frontend && python3 app.py" C-m

tmux attach -t overseec