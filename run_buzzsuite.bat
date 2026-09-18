@echo off
setlocal
cd /d "%~dp0"

echo.
echo ################################################################
echo #                                                              #
echo #                         BUZZSUITE                            #
echo #                                                              #
echo #                Mosquito Behavioural-Assay Suite              #
echo #                                                              #
echo ################################################################
echo.

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%PY_EXE%" (
    echo Starting BuzzSuite with local virtual environment Python...
    "%PY_EXE%" run_gui.py
) else (
    where conda >nul 2>&1
    if %errorlevel%==0 (
        echo Starting BuzzSuite with conda environment buzzsuite_env...
        conda run -n buzzsuite_env python run_gui.py
    ) else (
        echo Starting BuzzSuite with system Python...
        python run_gui.py
    )
)

if errorlevel 1 (
    echo.
    echo BuzzSuite failed to start.
    exit /b 1
)

echo.
echo BuzzSuite closed.
exit /b 0
