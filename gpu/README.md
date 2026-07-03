# AgentTrace cuVS worker

This directory contains the optional NVIDIA backend used by the Java
`CuvsTrajectorySearchBackend`.

## Files

- `setup-wsl.sh`: idempotent Ubuntu prerequisite and virtual-environment setup
- `requirements-lock.txt`: exact tested Python/CUDA package versions
- `run-python-wsl.sh`: run Python commands inside the pinned GPU environment
- `run-worker-wsl.sh`: runtime library-path setup and worker launcher
- `worker.py`: localhost JSON API backed by exact or CAGRA cuVS cosine search

## Setup and run

From PowerShell in the repository root:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/setup-wsl.sh
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/run-worker-wsl.sh
```

The worker listens on `http://127.0.0.1:8765`. Keep that terminal open while
using `--backend cuvs` from the Java service, `report`, `evaluate`, or
`evaluate-recovery` commands.

## Worker contract

- `GET /health`: cuVS version and index status
- `GET /stats`: trajectory count, vector dimension, and backend name
- `POST /rebuild`: replace the GPU index; accepts `brute_force` or `cagra`
- `POST /search`: filtered cosine-similarity search
- `POST /search/batch`: group compatible filters and search query matrices in one
  GPU call
- `POST /deduplicate`: GPU candidate generation plus exact cosine verification

The worker has no authentication and is intentionally localhost-only. The
exact backend provides ground truth and parity checks; CAGRA provides the
approximate production path. Filter subsets smaller than 128 vectors fall back
to exact search.
