@echo off
REM setup_calving.bat

set "ENV_NAME=calving"
set "PYTHON_VER=3.9"
set "REPO_URL=https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them.git"
set "REPO_DIR=Calving_Fronts_and_Where_to_Find_Them"

echo === Creazione ambiente conda %ENV_NAME% (Python %PYTHON_VER%)
call conda create -y -n %ENV_NAME% python=%PYTHON_VER%

echo === Attivazione ambiente
call conda activate %ENV_NAME%

echo === Aggiornamento pip
python -m pip install --upgrade pip

echo === Installazione PyTorch 1.9.0 + CUDA 11.1
pip install "torch==1.9.0+cu111; torchvision==0.10.0+cu111; torchaudio==0.9.0" -f https://download.pytorch.org/whl/torch_stable.html

echo === Blocco setuptools < 58
pip install "setuptools<58"

echo === Clonazione repository
git clone %REPO_URL%
cd %REPO_DIR%

echo === Installazione pacchetti da requirements.txt
pip install -r requirements.txt

echo === Rimozione repository clonato
cd ..
rd /s /q "%REPO_DIR%"

echo.
echo Setup completato!
echo Per usare l’ambiente: conda activate %ENV_NAME%
pause