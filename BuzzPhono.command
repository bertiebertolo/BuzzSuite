#!/bin/bash
# Double-click to open the BuzzPhono app in your browser (macOS; also runs on Linux).
PY="${BUZZPHONO_PYTHON:-}"
if [ -z "$PY" ]; then
  for r in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/anaconda3 /opt/miniconda3 /opt/homebrew/Caskroom/miniforge/base; do
    if [ -x "$r/envs/buzzphono_env/bin/python" ]; then PY="$r/envs/buzzphono_env/bin/python"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "BuzzPhono's environment was not found."
  echo "Install it once with:  conda env create -f environment-buzzphono.yml"
  echo "or set BUZZPHONO_PYTHON to its python."
  read -r -p "Press Enter to close."
  exit 1
fi
unset PYTHONPATH
exec "$PY" -m buzzphono app
