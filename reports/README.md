# Reproducible reports

- `minilm-embedding-run.json`: local ARM64 Java/ONNX embedding throughput
- `hash-evaluation.json`: 256-dimensional feature-hashing baseline
- `minilm-evaluation.json`: 384-dimensional MiniLM retrieval and dedup results
- `recovery-evaluation.json`: parent-excluded failed-trajectory recovery

The label set and methodology are documented in `docs/EVALUATION.md`.
Performance timings describe one development Mac run and are not
cross-hardware benchmarks.
