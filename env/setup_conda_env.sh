#!/usr/bin/env bash
# setup_calving.sh
# Script per creare l'ambiente Anaconda e installare le dipendenze
# del repository Calving_Fronts_and_Where_to_Find_Them.

set -e

# Parametri
ENV_NAME="FrostyCalvNet"
PYTHON_VER="3.9"
REPO_URL="https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them.git"
REPO_DIR="Calving_Fronts_and_Where_to_Find_Them"

echo ">>> Creazione dell'ambiente conda '${ENV_NAME}' (Python ${PYTHON_VER})"
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

# Attiva conda in modo non‑interattivo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">>> Aggiornamento pip"
python -m pip install --upgrade pip

echo ">>> Installazione PyTorch 1.9.0 + CUDA 11.1"
pip install "torch==1.9.0+cu111" -f https://download.pytorch.org/whl/torch_stable.html

echo ">>> Blocca setuptools < 58 (necessario per fil-finder==1.7)"
pip install "setuptools<58"

echo ">>> Clonazione del repository"
git clone "$REPO_URL"
cd "$REPO_DIR"

echo ">>> Installazione pacchetti da requirements.txt"
pip install -r requirements.txt

echo
echo "Setup completato!"
echo "Per usare l'ambiente: 'conda activate ${ENV_NAME}'"
echo "Esegui il codice dal path: $(pwd)"
