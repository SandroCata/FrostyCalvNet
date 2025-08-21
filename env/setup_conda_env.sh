#!/usr/bin/env bash
# setup_conda_env.sh
# Script to create conda env and install dependencies
# of Calving_Fronts_and_Where_to_Find_Them repository.

set -e

# Parameters
ENV_NAME="FrostyCalvNet"
PYTHON_VER="3.9"
REPO_URL="https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them.git"
REPO_DIR="Calving_Fronts_and_Where_to_Find_Them"

echo ">>> Creating conda environment: '${ENV_NAME}' (Python ${PYTHON_VER})"
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">>> Updating pip"
python -m pip install --upgrade pip

echo ">>> Installing PyTorch 1.9.0 + CUDA 11.1"
pip install "torch==1.9.0+cu111" -f https://download.pytorch.org/whl/torch_stable.html

echo ">>> Blocking setuptools < 58 (needed for fil-finder==1.7)"
pip install "setuptools<58"

echo ">>> Cloning the original repository"
git clone "$REPO_URL"
cd "$REPO_DIR"

echo ">>> Installing packets from requirements.txt"
pip install -r requirements.txt

echo ">>> Removing cloned repository"
cd ..
rm -rf "$REPO_DIR"

echo
echo "Setup completed!"
echo "To use the environment: 'conda activate ${ENV_NAME}'"
echo "Execute code from path: $(pwd)"
