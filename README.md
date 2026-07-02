# AgentTrace

AgentTrace is a Java/Lucene system for searching, inspecting, and deduplicating
GUI-agent training trajectories.

## Motivation

Collecting screenshots and actions is only the first half of building a GUI
agent dataset. As the collection grows, data engineers need to answer:

- Have we already collected this screen or task?
- Which successful trajectories resemble this failed trajectory?
- Which records are near-duplicates and overrepresented?
- Can we filter similar trajectories by platform and outcome?

AgentTrace turns each trajectory into an embedding, indexes it with Lucene
HNSW, and exposes search and deduplication operations through a small HTTP API.
The GPU is an optional scale-up backend, not the reason the product exists.

## Current MVP

- Java 21
- Apache Lucene HNSW vector index
- Local MiniLM semantic embeddings through Java ONNX Runtime
- BERT-compatible WordPiece tokenization implemented in Java
- Human-reviewed retrieval and deduplication evaluation
- Controlled failed-trajectory recovery with parent-excluded metrics
- Cosine-similarity search
- Platform and success/failure filters
- Near-duplicate grouping with exact cosine verification
- JSON HTTP API using Java virtual threads
- Unit tests and a tiny synthetic fixture
- Backend interface ready for a future NVIDIA cuVS implementation

The repository includes two versions of the same 500 public mobile-navigation
trajectories: a dependency-free feature-hashing baseline and a 384-dimensional
MiniLM semantic version. No screenshots are downloaded.

## Repository contents

- `src/`: Java implementation and tests
- `sample-data/`: 500 real trajectories plus controlled failure variants
- `labels/`: human-reviewed intent labels and failure ground truth
- `reports/`: committed machine-readable CPU results
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
                         |
                         v
               Lucene HNSW index
                  /            \
                 v              v
          similarity search   dedup groups
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

The importer reads the Hugging Face row API in pages of 100, discards signed
image URLs, and stores only normalized trajectory content and image counts.

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.1.0-SNAPSHOT.jar \
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

## Generate semantic embeddings locally

Download the pinned model and vocabulary described in
[`models/README.md`](models/README.md), then run:

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.1.0-SNAPSHOT.jar \
  embed \
  --input sample-data/aguvis-500.json \
  --output sample-data/aguvis-500-minilm.json \
  --model models/all-MiniLM-L6-v2/model_qint8_arm64.onnx \
  --vocab models/all-MiniLM-L6-v2/vocab.txt \
  --max-length 256 \
  --batch-size 16 \
  --report reports/aguvis-500-minilm-embedding.json
```

The embedding text combines the task instruction, app, and action sequence.
MiniLM produces a mean-pooled, L2-normalized 384-dimensional vector. On the
development Mac, all 500 trajectories were embedded in 14.7 seconds.

Generate a reproducible dataset/search smoke report:

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.1.0-SNAPSHOT.jar \
  report \
  --data sample-data/aguvis-500-minilm.json \
  --index data/aguvis-500-minilm-report-index \
  --threshold 0.92 \
  --candidate-k 10 \
  --output reports/aguvis-500-minilm-report.json
```

## Run

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.1.0-SNAPSHOT.jar \
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

## Next milestones

1. Transfer the project to the RTX 4070 Super Windows/WSL2 host.
2. Add a cuVS backend behind `TrajectorySearchBackend`.
3. Benchmark Lucene CPU versus cuVS GPU at 10K, 100K, and 1M vectors.
4. Add screenshot embeddings and multimodal fusion after backend isolation.

See [GPU migration](docs/GPU_MIGRATION.md) and
[dataset plan](docs/DATASET_PLAN.md).
