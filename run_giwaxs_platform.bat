@echo off
REM run_giwaxs_platform.bat
REM
REM Double-click launcher for Windows.
REM On first run: creates a local virtual environment (.venv) inside this
REM folder and installs the required packages -- no admin rights needed.
REM On later runs: just activates that environment and launches the platform.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo First run detected -- setting up a local Python environment...
    echo ^(This only happens once; it will be much faster next time.^)
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Starting GIWAXS Processing Platform...
echo.
python giwaxs_platform.py %*

echo.
pause
