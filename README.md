# AgentTrace

AgentTrace is a Java system for searching, inspecting, and deduplicating
GUI-agent training trajectories with Lucene CPU and NVIDIA cuVS GPU backends.

**Current release:** `v0.2.0`

**Status:** reproducible Java build, published CPU/GPU benchmark, and passing
GitHub Actions CI

## Project snapshot

| Area | Implementation |
|---|---|
| Product problem | Search, failure recovery, filtering, and deduplication for GUI-agent training trajectories |
| Java stack | Java 21, virtual threads, Maven, Jackson, ONNX Runtime, Java HTTP API |
| CPU search | Apache Lucene HNSW |
| GPU search | NVIDIA cuVS exact brute force and CAGRA |
| GPU integration | Batched Java client to a localhost Python/cuVS worker in Ubuntu/WSL2 |
| Embeddings | 384-dimensional MiniLM with Java WordPiece tokenization and mean pooling |
| Data | Public AGUVIS mobile-navigation trajectories; 500 checked-in records and a reproducible 10K benchmark pipeline |
| Evaluation | Retrieval, duplicate detection, parent-excluded failure recovery, Recall@K, throughput, and latency |
| Quality | Unit/integration tests, transport retries, resumable imports, machine-readable reports, and CI |

The central engineering question is not simply whether a GPU can run vector
search. It is whether one Java product interface can support Lucene and cuVS,
preserve retrieval quality, and expose the conditions under which GPU batching
actually improves end-to-end performance.

## Verified results

The primary benchmark used 10,000 real AGUVIS trajectories embedded with
MiniLM, 500 deterministic Top-10 queries, and exact cuVS as ground truth. It
ran on Windows 11 with an NVIDIA RTX 4070 SUPER through Ubuntu/WSL2 and cuVS
26.06.

| Backend | Index build | Mean Recall@10 | Batched queries/s | Individual p50 |
|---|---:|---:|---:|---:|
| cuVS exact | 1.165 s | 1.0000 | 5,113 | 4.32 ms |
| Lucene HNSW | 2.436 s | 0.9968 | 3,060 | 0.26 ms |
| cuVS CAGRA | 1.436 s | 1.0000 | 5,352 | 6.58 ms |

CAGRA delivered approximately **1.75x the warmed batched throughput** of
Lucene at perfect Recall@10 on this workload. Lucene remained much faster for
individual requests because it runs in-process and avoids the localhost/WSL2
boundary. The comparison therefore demonstrates a batching tradeoff, not a
claim that GPU search is universally faster.

Each timed batch follows an untimed same-shape warm-up. GPU timings include
JSON serialization and localhost transport. Full results are committed in
[`reports/windows-cpu-gpu-10000.json`](reports/windows-cpu-gpu-10000.json);
the 10K MiniLM run embedded 54.4 trajectories/s and is recorded in
[`reports/windows-minilm-10000-embedding-run.json`](reports/windows-minilm-10000-embedding-run.json).

## Engineering highlights

- Defined a `TrajectorySearchBackend` abstraction with runtime-selectable
  Lucene, exact cuVS, and CAGRA implementations.
- Added batched query transport that groups compatible metadata filters and
  executes one GPU query matrix per group.
- Preserved a common cosine-search, filtering, statistics, and deduplication
  contract across CPU and GPU backends.
- Added exact cuVS ground truth and a reproducible benchmark harness for build
  time, Recall@K, full-recall rate, throughput, and p50/p95 latency.
- Built a storage-safe AGUVIS pipeline using Parquet column pruning so text and
  metadata can be imported without downloading screenshot bytes.
- Implemented Java-native MiniLM inference, BERT-compatible WordPiece
  tokenization, mean pooling, and L2 normalization.
- Hardened cross-language execution with bounded retries, import checkpoints,
  adaptive row-API page splitting, deterministic query selection, and pinned
  WSL2/cuVS dependencies.
- Added human-reviewed labels and controlled failure injection to test whether
  incomplete trajectories retrieve alternative successful traces rather than
  only their exact source.

## Resume and portfolio summary

The following wording is intentionally limited to work demonstrated by this
repository:

**AgentTrace - GPU-Accelerated Vector Search System**

*Java 21, Apache Lucene, NVIDIA cuVS, CAGRA, Python, WSL2, ONNX Runtime*

- Built a Java 21 vector-search and deduplication system integrating Lucene
  HNSW with NVIDIA cuVS exact and CAGRA backends through a batched
  Java-to-Python GPU execution path.
- Benchmarked 10,000 real GUI-agent trajectories across 500 Top-10 queries;
  achieved 1.0000 Recall@10 and 5.35K queries/s with CAGRA, approximately
  1.75x the warmed batched throughput of Lucene HNSW.
- Engineered a reproducible data and reliability pipeline with local MiniLM
  inference, storage-safe Parquet ingestion, metadata filtering, transport
  retries, Maven tests, versioned reports, and GitHub Actions CI.

Useful keywords for accurate project indexing: `Java`, `JVM`, `Lucene`,
`HNSW`, `NVIDIA cuVS`, `CAGRA`, `approximate nearest neighbor search`,
`vector search systems`, `GPU acceleration`, `performance benchmarking`,
`cross-language debugging`, `Python`, `WSL2`, `ONNX Runtime`, and `CI/CD`.

## Scope and implementation boundaries

- AgentTrace integrates NVIDIA cuVS; it does **not** implement custom CUDA or
  C++ kernels.
- The current GPU path is Java HTTP client -> localhost Python worker -> cuVS
  in WSL2. It is not yet a direct JNI or Foreign Function and Memory API
  binding from the JVM.
- The published scale point is 10K real vectors. The 100K and 1M runs remain
  future work.
- Batch and individual-query results are reported separately because transport
  overhead materially changes the CPU/GPU comparison.
- The 57 injected failures are controlled evaluation cases, not naturally
  collected production failures.

## Motivation

Collecting screenshots and actions is only the first half of building a GUI
agent dataset. As the collection grows, data engineers need to answer:

- Have we already collected this screen or task?
- Which successful trajectories resemble this failed trajectory?
- Which records are near-duplicates and overrepresented?
- Can we filter similar trajectories by platform and outcome?

AgentTrace turns each trajectory into an embedding, indexes it with a selectable
search backend, and exposes search and deduplication operations through a small
HTTP API. The GPU is an optional scale-up backend, not the reason the product
exists.

## Implemented in v0.2.0

- Java 21
- Apache Lucene HNSW vector index
- Exact and CAGRA NVIDIA cuVS cosine search through a localhost WSL2 worker
- Batched GPU query transport
- Local MiniLM semantic embeddings through Java ONNX Runtime
- BERT-compatible WordPiece tokenization implemented in Java
- Human-reviewed retrieval and deduplication evaluation
- Controlled failed-trajectory recovery with parent-excluded metrics
- Cosine-similarity search
- Platform and success/failure filters
- Near-duplicate grouping with exact cosine verification
- JSON HTTP API using Java virtual threads
- Unit tests and a tiny synthetic fixture
- Runtime-selectable `lucene`, exact `cuvs`, and `cuvs-cagra` backends

The repository includes two versions of the same 500 public mobile-navigation
trajectories: a dependency-free feature-hashing baseline and a 384-dimensional
MiniLM semantic version. No screenshots are downloaded.

## Repository contents

- `src/`: Java implementation and tests
- `gpu/`: pinned WSL2/cuVS setup, launcher, and worker
- `tools/`: text-only AGUVIS Parquet acquisition tooling
- `sample-data/`: 500 real trajectories plus controlled failure variants
- `labels/`: human-reviewed intent labels and failure ground truth
- `reports/`: committed machine-readable CPU and GPU results
- `CHANGELOG.md`: release history
- `docs/EVALUATION.md`: evaluation methodology and limitations
- `docs/GITHUB_SETUP.md`: GitHub publishing and Windows/WSL2 handoff
- `models/README.md`: pinned model download and checksum instructions

Third-party dataset and model terms are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Architecture

```text
trajectory JSON
  instruction + platform + app + actions + embedding
                         |
                         v
            TrajectorySearchBackend
                 /                 \
                v                   v
       Lucene HNSW index     batched cuVS client (Java)
                                     |
                                     v
                    WSL2 exact/CAGRA GPU worker
                 \                 /
                  v               v
             similarity search + dedup groups
                         |
                         v
                    HTTP API
```

The backend boundary is deliberate. Lucene provides the portable in-process
baseline. The cuVS worker isolates Linux-only CUDA dependencies while keeping
the Java API stable, making correctness and transport costs directly
measurable. A direct JVM-native cuVS integration is a logical future
optimization rather than something the current architecture silently assumes.

## Build

Requirements:

- JDK 21

On Apple Silicon, use an ARM64 JDK. An Intel JDK running through Rosetta cannot
load the ARM64 ONNX Runtime library.

```bash
./mvnw test
./mvnw package
```

On Windows, use `mvnw.cmd test` and `mvnw.cmd package`.

## Import a storage-safe AGUVIS sample

The importer reads the Hugging Face row API with resumable checkpoints,
discards signed image URLs, and stores only normalized trajectory content and
image counts.

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  import-aguvis \
  --config mobile.navigation \
  --split train \
  --limit 500 \
  --dimension 256 \
  --output sample-data/aguvis-500.json
```

The current vectors use deterministic feature hashing over instructions and
action descriptions. This is a dependency-free pipeline baseline, not a
semantic ML embedding.

For larger text-only samples, use Parquet column pruning so screenshot bytes
are never downloaded. PyArrow 19.0.0 has a known nested-Parquet regression;
use 19.0.1 or newer:

```bash
python -m pip install "pyarrow>=19.0.1" fsspec requests
python tools/fetch_aguvis_parquet.py \
  --config mobile.navigation \
  --split train \
  --limit 10000 \
  --output work/aguvis-10000-rows.json

java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  import-aguvis \
  --rows-file work/aguvis-10000-rows.json \
  --limit 10000 \
  --dimension 256 \
  --output work/aguvis-10000.json
```

## Generate semantic embeddings locally

Download the pinned model and vocabulary described in
[`models/README.md`](models/README.md). The CLI selects the ARM64 model on
ARM64 systems and the AVX2 model on x64 systems, so `--model` can be omitted
when the model is stored at the documented path:

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  embed \
  --input sample-data/aguvis-500.json \
  --output sample-data/aguvis-500-minilm.json \
  --vocab models/all-MiniLM-L6-v2/vocab.txt \
  --max-length 256 \
  --batch-size 16 \
  --report work/local-minilm-embedding-run.json
```

The embedding text combines the task instruction, app, and action sequence.
MiniLM produces a mean-pooled, L2-normalized 384-dimensional vector. On the
development Mac, all 500 trajectories were embedded in 14.7 seconds.

Generate a reproducible dataset/search smoke report:

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  report \
  --data sample-data/aguvis-500-minilm.json \
  --index data/aguvis-500-minilm-report-index \
  --threshold 0.92 \
  --candidate-k 10 \
  --output reports/minilm-dataset-report.json
```

## Run

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  --data sample-data/trajectories.json \
  --index data/lucene-index \
  --port 8080
```

Health and index statistics:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/stats
```

Find successful mobile trajectories similar to the Wi-Fi example:

```bash
curl -X POST http://localhost:8080/api/search \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding": [1,0,0,0,0,0,0,0],
    "k": 3,
    "platform": "mobile",
    "app": "android-settings",
    "success": true
  }'
```

Find successful trajectories resembling a known failed trajectory:

```bash
curl -X POST http://localhost:8080/api/search/by-trajectory \
  -H 'Content-Type: application/json' \
  -d '{
    "trajectoryId": "wifi-failed-001",
    "k": 3,
    "success": true
  }'
```

Find near-duplicate groups:

```bash
curl -X POST http://localhost:8080/api/deduplicate \
  -H 'Content-Type: application/json' \
  -d '{"threshold": 0.985, "candidateK": 5}'
```

## Run with NVIDIA cuVS

The GPU backend uses pinned cuVS 26.06 packages inside Ubuntu/WSL2. From
PowerShell, install the isolated environment once:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/setup-wsl.sh
```

Start the localhost worker in one terminal:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/run-worker-wsl.sh
```

Then select cuVS from the Java service or evaluation commands:

```powershell
java --add-modules jdk.incubator.vector `
  --enable-native-access=ALL-UNNAMED `
  -jar target\agenttrace-0.2.0.jar `
  --backend cuvs `
  --cuvs-algorithm cagra `
  --cuvs-url http://127.0.0.1:8765 `
  --data sample-data\trajectories.json `
  --port 8080
```

The worker binds to `127.0.0.1` by default. See
[`docs/GPU_MIGRATION.md`](docs/GPU_MIGRATION.md) for setup, evaluation, and
current limitations.

## What makes this a real project

The output is not merely a CPU/GPU benchmark. A data engineer can use the
system to retrieve comparable successful traces, inspect common failure modes,
and reduce duplicate training examples. The committed performance and recall
measurements validate the current 10K scale point and make the remaining scale
work explicit.

The current labeled CPU evaluation shows MiniLM improving Recall@5 from 0.460
to 0.608 and MRR from 0.836 to 0.965 over feature hashing. In a stricter
failed-trajectory test that excludes each exact parent, all 57 queries retrieve
another same-intent successful trajectory in the Top 5. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) for methodology and limitations.

The native Windows verification is also complete. On the development Windows
11 x64 host, the pinned AVX2 MiniLM model embedded all 500 trajectories in 9.7
seconds, the full Maven build passed, and the HTTP API smoke test passed. The
machine-readable results are in `reports/windows-minilm-embedding-run.json`,
`reports/windows-minilm-evaluation.json`, and
`reports/minilm-dataset-report.json`.

The exact cuVS backend reproduces the Lucene Recall@5, MRR, recovery metrics,
and all six duplicate groups on the same committed vectors. The CAGRA path uses
the same filters and result contract, while `/search/batch` amortizes
localhost/WSL2 transport across query sets.

On the 10,000-vector MiniLM run with 500 Top-10 queries, CAGRA reached `1.0000`
Recall@10 at `5,352` batched queries/s. Lucene HNSW reached `0.9968` Recall@10
at `3,060` queries/s. CAGRA was about 1.75x faster for the warmed batch, while
Lucene remained much faster for individual queries because it avoids
localhost/WSL2 transport. See `reports/windows-cpu-gpu-10000.json`.

## Next milestones

1. Extend the real-vector benchmark to 100K and 1M vectors.
2. Record GPU memory and power alongside throughput and latency.
3. Add screenshot embeddings and multimodal fusion.

See [GPU migration](docs/GPU_MIGRATION.md) and
[dataset plan](docs/DATASET_PLAN.md).

## License

AgentTrace source code is licensed under the
[Apache License 2.0](LICENSE). Dataset, model, and dependency terms remain
separate as described in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
