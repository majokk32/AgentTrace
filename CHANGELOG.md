# Changelog

## 0.2.0 - 2026-07-02

- Added batched query transport for the localhost cuVS worker.
- Added selectable exact brute-force and approximate CAGRA GPU indexes.
- Added a matched-recall CPU/GPU benchmark command with build, throughput,
  latency, and Recall@K metrics.
- Added a text-only Parquet importer for storage-safe 10K AGUVIS runs.
- Added resumable row-API checkpoints, bounded page splitting, and cuVS
  transport retries for long-running local workflows.

## 0.1.0 - 2026-07-02

- Added Java 21 trajectory search, filtering, and deduplication HTTP APIs.
- Added Lucene HNSW and exact NVIDIA cuVS GPU backends.
- Added local MiniLM ONNX embedding with architecture-aware model selection.
- Added human-reviewed retrieval, duplicate, and failure-recovery evaluations.
- Added reproducible Windows, Lucene, and cuVS reports.
- Added pinned Ubuntu/WSL2 cuVS setup and localhost worker tooling.
- Added Apache-2.0 source licensing and continuous integration.
