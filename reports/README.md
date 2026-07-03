# Reproducible reports

- `minilm-embedding-run.json`: local ARM64 Java/ONNX embedding throughput
- `windows-minilm-embedding-run.json`: local x64 AVX2 Java/ONNX throughput
- `minilm-dataset-report.json`: Windows Lucene dataset/search smoke report
- `hash-evaluation.json`: 256-dimensional feature-hashing baseline
- `minilm-evaluation.json`: 384-dimensional MiniLM retrieval and dedup results
- `windows-minilm-evaluation.json`: Windows Lucene evaluation of committed vectors
- `recovery-evaluation.json`: parent-excluded failed-trajectory recovery
- `cuvs-minilm-evaluation.json`: matched exact-cuVS retrieval evaluation
- `cuvs-minilm-dataset-report.json`: exact-cuVS search and dedup smoke report
- `cuvs-recovery-evaluation.json`: exact-cuVS failed-trajectory recovery
- `windows-minilm-10000-embedding-run.json`: 10K x64 AVX2 embedding throughput
- `windows-cpu-gpu-10000.json`: 10K Lucene, exact-cuVS, and CAGRA benchmark

The label set and methodology are documented in `docs/EVALUATION.md`.
Timings are single-machine development runs, not cross-hardware benchmarks.
