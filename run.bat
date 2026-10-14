@echo off
cd /d "%~dp0"
if defined VIRTUAL_ENV goto active
if defined CONDA_PREFIX goto active
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py %*
  exit /b
)
:active
python app.py %*
exit /b
