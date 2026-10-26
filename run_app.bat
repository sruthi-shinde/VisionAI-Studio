@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the virtual environment first using:
    echo py -3.13 -m venv .venv
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
