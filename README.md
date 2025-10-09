# overseec

Overseec is an application that leverages a VLLM-powered backend for large language model inference and a Python frontend for user interaction. This guide provides instructions for setting up and running the project using either Conda or Docker.

## Installation

You can set up the project using either Conda for a local installation or Docker for a containerized environment.

### Conda

1. **Create and activate the Conda environment:**
   ```bash
   conda create -n <env_name> python=3.10 -y
   conda activate <env_name>
   ```

2. **Install dependencies:**
   ```bash
   # Install VLLM for CUDA 12.8
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

   # Install other requirements
   pip install FastGeodis --no-build-isolation
   pip install -r requirements.txt
   ```

3. **Download model checkpoints:**
   ```bash
   mkdir -p checkpoints
   cd checkpoints

   # ViT-H (default)
   curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

   # ViT-L
   curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

   # ViT-B (smallest)
   curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

   cd ..
   ```

4. **Make the startup script executable:**
   ```bash
   chmod +x start_overseec.sh
   ```

### Docker

1. **Build the Docker image:**
   ```bash
   docker build -t overseec-cu128 .
   ```

2. **Run the Docker container:**
   This command starts the container, maps the necessary ports and volumes, and provides access to the host's GPUs.
   ```bash
    chmod +x start_docker.sh
    ./start_docker.sh
   ```

## Usage

### Interactive
The `start_overseec.sh` script automates the setup process by launching the VLLM server and the frontend application in a `tmux` session.

#### Script Arguments
The script is configured using the following command-line arguments. The order of arguments does not matter.

| Argument           | Description                                                                                                                           | Example                                     |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------ |
| **`--vllm-model`** | Specifies the language model for the VLLM server. If omitted, the server will show an interactive menu to choose a model.              | `--vllm-model gemma-2-27b-it`               |
| **`--env`**        | Sets the Conda environment name to activate. The default is `vllm`. Use `--env no_conda` to run without Conda **(required for Docker)**.   | `--env my_overseec_env`                     |
| **`--dir`**        | Specifies the project's root directory. Defaults to the current directory where the script is run.                                    | `--dir /home/user/overseec`                 |
| **`--vllm-device`**| Sets the `CUDA_VISIBLE_DEVICES` for the VLLM server to control which GPU(s) are used. Can be a single ID or a comma-separated list.    | `--vllm-device "0,1"`                      |

#### Command Examples

**1. Basic Usage (Conda Installation)**
This command activates your specified Conda environment and starts the services, prompting you to select an LLM interactively.

```bash
./start_overseec.sh --env <your_env_name>
```

**2. Launching with a Specific Model and GPU (Conda)**
This is useful for automation or if you know exactly which model and GPU you want to use.

```bash
./start_overseec.sh --env <your_env_name> --vllm-model Qwen2.5-Coder-14B-Instruct --vllm-device "0"
```

**3. Running Inside the Docker Container**
When you are inside the Docker container, the environment is already set up, so you should disable Conda activation.

```bash
# This is the recommended command to run after starting the Docker container
# This will start the interactive environment to choose the vllm model
./start_overseec.sh --env no_conda

# This is for running a specific vllm model on a device in docker 
./start_overseec.sh --env no_conda --vllm-model Qwen2.5-Coder-14B-Instruct --vllm-device "1,2"
```

### Docker Compose
A Docker Compose configuration is provided for automatic web hosting using https and a reverse proxy via Caddy.

To use this setup, you must create and define a `.env` file in the root directory of the repo with the following line:
```
SITE_HOSTNAME=<your.server.domain.name>
```

Then running OVerSeeC and Caddy is accomplished via
```
$ cd /path/to/overseec/repo
$ docker compose up -d
```