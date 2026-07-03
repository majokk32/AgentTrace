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
      cuVS exact or CAGRA cosine index on the NVIDIA GPU
```

The worker keeps metadata on the host and vectors on the GPU. It caches cuVS
indexes for metadata-filter combinations, batches queries with matching
filters, supports bulk duplicate candidate generation, and verifies duplicate
thresholds with exact cosine.

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
  -jar target\agenttrace-0.2.0.jar `
  evaluate `
  --backend cuvs `
  --cuvs-algorithm brute_force `
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

Exact cuVS remains the ground-truth path. Select approximate CAGRA with
`--cuvs-algorithm cagra` (or `--backend cuvs-cagra`). Batched searches use one
localhost request and one GPU query matrix per compatible metadata filter.

Run the matched-recall benchmark:

```powershell
java --add-modules jdk.incubator.vector `
  --enable-native-access=ALL-UNNAMED `
  -jar target\agenttrace-0.2.0.jar `
  benchmark `
  --data work\aguvis-10000-minilm.json `
  --index data\benchmark-10000-lucene-index `
  --queries 500 `
  --k 10 `
  --output reports\windows-cpu-gpu-10000.json
```

The report records build time, batched throughput, single-query p50/p95
latency, Recall@K, full-recall rate, and whether each returned top result is in
the exact ground-truth set. Batch timings include JSON and localhost WSL2
transport. Remaining scale work is 100K/1M vectors plus GPU memory and power
telemetry.

## Verified 10K result

The Windows RTX 4070 Super run used 10,000 real mobile-navigation MiniLM
vectors and 500 deterministic Top-10 queries:

| Backend | Recall@10 | Batched queries/s | Individual p50 |
|---|---:|---:|---:|
| cuVS exact | 1.0000 | 5,113 | 4.32 ms |
| Lucene HNSW | 0.9968 | 3,060 | 0.26 ms |
| cuVS CAGRA | 1.0000 | 5,352 | 6.58 ms |

Each timed batch follows an untimed same-shape warm-up. CAGRA is about 1.75x
faster than Lucene for this batch, but Lucene is substantially faster for
single requests because it has no HTTP/WSL2 boundary.
