@echo off
rem Create buzzsuite_env on a new Windows PC, matching the machine tracking was validated on, then
rem run check_install.py. Run it from "Anaconda Prompt" (Start menu), in this folder.
setlocal
cd /d "%~dp0"

where conda >nul 2>&1
if errorlevel 1 (
    echo conda was not found. Open "Anaconda Prompt" from the Start menu and run this file from there.
    pause
    exit /b 1
)

call conda env list | findstr /b /c:"buzzsuite_env " >nul
if not errorlevel 1 (
    echo buzzsuite_env already exists - checking it. To rebuild it from scratch, first run:
    echo     conda env remove -n buzzsuite_env
    goto check
)

call conda env create -f environment.yml
if errorlevel 1 (
    echo Creating buzzsuite_env failed.
    pause
    exit /b 1
)

rem PyPI wheels for numpy/matplotlib: the conda-forge builds crash every plot on PCs with
rem Memory Integrity + CET (DEVLOG 2026-09-16).
call conda run -n buzzsuite_env python -m pip install --force-reinstall --no-deps numpy==1.24.4 matplotlib==3.7.3 kiwisolver==1.4.5 contourpy==1.1.1
if errorlevel 1 (
    echo Installing the numpy/matplotlib wheels failed.
    pause
    exit /b 1
)

:check
call conda run -n buzzsuite_env --no-capture-output python check_install.py
pause
