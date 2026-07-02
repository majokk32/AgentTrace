# AgentTrace cuVS worker

This directory contains the optional NVIDIA backend used by the Java
`CuvsTrajectorySearchBackend`.

## Files

- `setup-wsl.sh`: idempotent Ubuntu prerequisite and virtual-environment setup
- `requirements-lock.txt`: exact tested Python/CUDA package versions
- `run-worker-wsl.sh`: runtime library-path setup and worker launcher
- `worker.py`: localhost JSON API backed by exact cuVS cosine search

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
- `POST /rebuild`: replace the GPU index with a trajectory JSON array
- `POST /search`: filtered cosine-similarity search
- `POST /deduplicate`: GPU candidate generation plus exact cosine verification

The worker has no authentication and is intentionally localhost-only. The
current backend uses exact brute-force cuVS search to establish semantic parity
with Lucene before adding approximate CAGRA search and large-scale tuning.
