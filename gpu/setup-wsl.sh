#!/usr/bin/env bash
set -euo pipefail

venv_path="${AGENTTRACE_CUVS_VENV:-/opt/agenttrace-cuvs}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! dpkg-query -W -f='${Status}' python3-venv 2>/dev/null \
    | grep -q "ok installed"; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends python3-venv ca-certificates
fi

python3 -m venv "$venv_path"
"$venv_path/bin/python" -m pip install --upgrade "pip==26.1.2"
"$venv_path/bin/python" -m pip install \
  --requirement "$script_dir/requirements-lock.txt" \
  --extra-index-url=https://pypi.nvidia.com

site_packages="$("$venv_path/bin/python" -c \
  "import site; print(site.getsitepackages()[0])")"
library_path="$(find "$site_packages" -type d \
  \( -name lib -o -name lib64 \) -print | paste -sd: -)"

LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$venv_path/bin/python" -c \
  "import cuvs; from cuvs.neighbors import brute_force; print(f'cuVS {cuvs.__version__} installed in $venv_path')"
