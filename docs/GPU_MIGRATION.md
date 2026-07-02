# GPU backend on Windows/WSL2

The Java/Lucene backend remains the portable baseline. The NVIDIA backend is
implemented behind the same `TrajectorySearchBackend` interface through a
localhost-only Python worker running cuVS inside WSL2.

## Verified environment

- Windows 11 and JDK 21
- NVIDIA RTX 4070 Super with driver 591.86
- WSL2 with Ubuntu 24.04.4 LTS
- cuVS 26.06.00 with the CUDA 12.9 Python runtime

cuVS prebuilt packages are Linux-only. WSL2 is therefore the boundary between
the Java service and the GPU runtime.

## Architecture

```text
Windows Java service
  |
  +-- LuceneTrajectorySearchBackend
  |
  +-- CuvsTrajectorySearchBackend
          |
          | localhost JSON HTTP
          v
      Ubuntu/WSL2 worker
          |
          v
      cuVS brute-force cosine index on the NVIDIA GPU
```

The worker keeps metadata on the host and vectors on the GPU. It caches exact
cuVS indexes for metadata-filter combinations, supports bulk duplicate
candidate generation, and verifies duplicate thresholds with exact cosine.

## Install

Install Ubuntu 24.04 and verify passthrough:

```powershell
wsl.exe --install -d Ubuntu-24.04
wsl.exe -d Ubuntu-24.04 -- nvidia-smi
```

Create the pinned environment:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/setup-wsl.sh
```

The script installs `cuvs-cu12==26.6.0` under `/opt/agenttrace-cuvs`. Generated
environments, model weights, and indexes remain outside Git.

## Run

Start the worker:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/run-worker-wsl.sh
```

Run the labeled evaluation from another PowerShell terminal:

```powershell
java --add-modules jdk.incubator.vector `
  --enable-native-access=ALL-UNNAMED `
  -jar target\agenttrace-0.1.0-SNAPSHOT.jar `
  evaluate `
  --backend cuvs `
  --cuvs-url http://127.0.0.1:8765 `
  --data sample-data\aguvis-500-minilm.json `
  --labels labels\aguvis-500-intents-v1.json `
  --index data\unused-cuvs-index `
  --embedding-model all-MiniLM-L6-v2 `
  --k 5 `
  --duplicate-threshold 0.92 `
  --output reports\cuvs-minilm-evaluation.json
```

## Verified parity

On the checked-in 500-vector MiniLM dataset, Lucene and exact cuVS produce the
same Recall@5 (`0.608`), MRR (`0.965`), and six duplicate groups. The controlled
failure evaluation also matches: parent-excluded HitRate@5 is `1.000` and MRR
is `0.918`.

The current cuVS worker performs one localhost request per search. At only 500
vectors that transport overhead dominates, so the committed timing is a
correctness baseline rather than evidence of acceleration.

## Remaining scale work

- Add batched query transport so GPU measurements exclude per-query HTTP cost.
- Add cuVS CAGRA for approximate search and tune it at matched recall.
- Benchmark 10K, 100K, and 1M real vectors.
- Record throughput and GPU memory alongside build time and p50/p95 latency.
