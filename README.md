# AgentTrace

AgentTrace is a Java system for searching, inspecting, and deduplicating
GUI-agent training trajectories with Lucene CPU and NVIDIA cuVS GPU backends.

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

## Current MVP

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
and reduce duplicate training examples. Performance and recall measurements
will later validate whether the implementation scales.

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
