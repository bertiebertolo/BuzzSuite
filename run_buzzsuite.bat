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
if exist "%PY_EXE%" goto run_direct

where conda >nul 2>&1
if not errorlevel 1 goto run_conda

rem conda is not on PATH (the default for a new Anaconda install): look for the env's Python directly.
for %%R in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "C:\anaconda3" "C:\miniconda3") do (
    if exist "%%~R\envs\buzzsuite_env\python.exe" (
        set "PY_EXE=%%~R\envs\buzzsuite_env\python.exe"
        goto run_direct
    )
)

echo buzzsuite_env was not found. Starting BuzzSuite with system Python...
python run_gui.py
goto done

:run_direct
echo Starting BuzzSuite with %PY_EXE%
"%PY_EXE%" run_gui.py
goto done

:run_conda
echo Starting BuzzSuite with conda environment buzzsuite_env...
call conda run -n buzzsuite_env python run_gui.py

:done
if errorlevel 1 (
    echo.
    echo BuzzSuite failed to start.
    pause
    exit /b 1
)

echo.
echo BuzzSuite closed.
exit /b 0
