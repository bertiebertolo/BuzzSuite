@echo off
rem Create buzzsuite_env on a new Windows PC, matching the machine tracking was validated on, run
rem check_install.py, then create buzzphono_env for BuzzPhono. Run it from "Anaconda Prompt"
rem (Start menu), in this folder. Safe to run again: environments that exist are only checked.
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

rem BuzzPhono runs in its own environment (Python 3.12); BuzzSuite's BuzzPhono tab starts it there.
call conda env list | findstr /b /c:"buzzphono_env " >nul
if not errorlevel 1 (
    echo buzzphono_env already exists - checking it.
    goto checkphono
)
echo Creating buzzphono_env for BuzzPhono. This takes a few minutes.
call conda env create -f environment-buzzphono.yml
if errorlevel 1 (
    echo Creating buzzphono_env failed. The BuzzSuite app works without it, but BuzzPhono will not open.
    pause
    exit /b 1
)

:checkphono
call conda run -n buzzphono_env --no-capture-output buzzphono --version
pause
