#!/usr/bin/env bash
set -euo pipefail

venv_path="${AGENTTRACE_CUVS_VENV:-/opt/agenttrace-cuvs}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
site_packages="$("$venv_path/bin/python" -c \
  "import site; print(site.getsitepackages()[0])")"
library_path="$(find "$site_packages" -type d \
  \( -name lib -o -name lib64 \) -print | paste -sd: -)"

export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$venv_path/bin/python" "$script_dir/worker.py" "$@"
