@echo off
rem PRISM-Endo one-click start: first run sets everything up (needs internet, ~10 min), later runs start in seconds.
cd /d "%~dp0"
title PRISM-Endo

where py >nul 2>nul && (set "PY=py -3.11") || (set "PY=python")
%PY% --version >nul 2>nul || (
  echo Python 3.11 is required. Install it from https://www.python.org/downloads/release/python-3119/
  echo and tick "Add python.exe to PATH" during setup, then run START.bat again.
  pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating the Python environment...
  %PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
  echo [2/3] Installing packages - first run only, this takes several minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install torch torchvision || (echo Package install failed - check the internet connection. & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed - check the internet connection. & pause & exit /b 1)
)

if not exist "outputs\models\instrument_seg_multivideo\best.pt" (
  echo Downloading the trained models - first run only, about 1 GB...
  curl.exe -L --fail -o prism-endo-models.zip https://github.com/SeptuaJananiB/Prism_Endo/releases/download/models-v1/prism-endo-models.zip || (echo Model download failed - check the internet connection. & pause & exit /b 1)
  tar -xf prism-endo-models.zip || (echo Could not unpack the models. & pause & exit /b 1)
  del prism-endo-models.zip
)

echo [3/3] Starting PRISM-Endo at http://localhost:8510  (close this window to stop)
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8510"
".venv\Scripts\python.exe" -m uvicorn src.api.server:app --port 8510 --host 127.0.0.1
pause
