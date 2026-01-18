#!/bin/bash

# Enable debug mode and exit on error
set -x
set -e

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com

# Install pre-built flash-attention
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

python setup.py develop

# We don't need all these, will clean up later
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
hf download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts

# Download our models
mkdir -p wan_models
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./wan_models/Wan2.1-I2V-14B-480P
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./wan_models/Wan2.1-T2V-1.3B
