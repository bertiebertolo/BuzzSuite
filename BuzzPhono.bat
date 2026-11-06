@echo off
rem Double-click to open the BuzzPhono app in your browser (Windows).
setlocal
set "PY=%BUZZPHONO_PYTHON%"
if not defined PY for %%R in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
  if not defined PY if exist "%%~R\envs\buzzphono_env\python.exe" set "PY=%%~R\envs\buzzphono_env\python.exe"
)
if not defined PY (
  echo BuzzPhono's environment was not found.
  echo Install it once with:  conda env create -f environment-buzzphono.yml
  echo or set BUZZPHONO_PYTHON to its python.exe.
  pause
  exit /b 1
)
set PYTHONPATH=
"%PY%" -m buzzphono app
if errorlevel 1 pause
