# Embedding and backend evaluation

The labeled evaluation answers two product questions:

1. Does a real semantic embedding improve retrieval over feature hashing?
2. Can an incomplete failed trajectory retrieve a comparable successful trace?

## Human-reviewed labels

`labels/aguvis-500-intents-v1.json` contains:

- 57 real AGUVIS trajectories
- 10 retrieval intent groups
- 44 duplicate-positive pairs
- 187 cross-intent duplicate-negative pairs
- 10 additional same-family hard negatives

Retrieval labels are intentionally broader than duplicate labels. Alarms at
different times, for example, share a retrieval family but are not duplicates
because their requested slot values differ.

## Hash versus MiniLM

Both embeddings use the same 500 trajectories, labels, Lucene backend, and
Top-5 evaluation:

| Metric | Feature hash | MiniLM |
|---|---:|---:|
| Recall@5 | 0.460 | 0.608 |
| Precision@5 | 0.642 | 0.779 |
| HitRate@5 | 0.912 | 1.000 |
| MRR | 0.836 | 0.965 |
| Best calibrated dedup F1 | 0.821 | 0.907 |

The deduplication threshold sweep covers 0.60 through 0.99. Its best result is
a calibration result on this label set, not an independent held-out test.

## Controlled failure recovery

`inject-failures` keeps the first two thirds of each labeled trajectory's
actions, marks the new record failed, embeds it independently, and records its
source trajectory plus other successful members of the same intent group.

The stricter recovery evaluation excludes the exact parent from the returned
ranking:

| Metric | MiniLM + Lucene |
|---|---:|
| Exact-parent Recall@1 | 0.930 |
| Exact-parent Recall@5 | 1.000 |
| Parent-excluded intent Recall@5 | 0.576 |
| Parent-excluded intent HitRate@5 | 1.000 |
| Parent-excluded intent MRR | 0.918 |

These 57 controlled failures verify the end-to-end retrieval workflow. They do
not replace naturally collected failures and should not be presented as such.

## Lucene and cuVS parity

The exact cuVS brute-force backend uses the same 500 MiniLM vectors and labels.
Its ranking metrics match Lucene:

| Metric | Lucene HNSW | cuVS exact |
|---|---:|---:|
| Recall@5 | 0.608 | 0.608 |
| Precision@5 | 0.779 | 0.779 |
| HitRate@5 | 1.000 | 1.000 |
| MRR | 0.965 | 0.965 |
| Duplicate groups at 0.92 | 6 | 6 |

The cuVS recovery results also match the Lucene values above. Exact cuVS is
also the ground truth for the 10K CAGRA benchmark. Batched GPU measurements use
one localhost request per query set; single-query measurements retain the
end-to-end HTTP/WSL2 product path.

## 10K approximate-search result

With 10,000 real mobile-navigation MiniLM vectors and 500 deterministic Top-10
queries, CAGRA achieved `1.0000` mean Recall@10 and a `1.0000` full-recall
rate against exact cuVS. Lucene achieved `0.9968` mean Recall@10 and a `0.976`
full-recall rate. Warmed batch throughput was `5,352` queries/s for CAGRA and
`3,060` for Lucene. These are single-machine development measurements, not a
general hardware comparison.

## Reproduce

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  evaluate \
  --data sample-data/aguvis-500-minilm.json \
  --labels labels/aguvis-500-intents-v1.json \
  --index data/eval-minilm-index \
  --embedding-model all-MiniLM-L6-v2-qint8-arm64 \
  --k 5 \
  --duplicate-threshold 0.92 \
  --output reports/minilm-evaluation.json
```

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  inject-failures \
  --input sample-data/aguvis-500-minilm.json \
  --output sample-data/aguvis-500-plus-57-labeled-failures.json \
  --pairs labels/aguvis-labeled-failure-pairs-57.json \
  --labels labels/aguvis-500-intents-v1.json \
  --vocab models/all-MiniLM-L6-v2/vocab.txt
```

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar target/agenttrace-0.2.0.jar \
  evaluate-recovery \
  --data sample-data/aguvis-500-plus-57-labeled-failures.json \
  --pairs labels/aguvis-labeled-failure-pairs-57.json \
  --index data/recovery-eval-index \
  --k 5 \
  --output reports/recovery-evaluation.json
```

With the worker from `docs/GPU_MIGRATION.md` running, add these options to
either evaluation command:

```text
--backend cuvs
--cuvs-url http://127.0.0.1:8765
--cuvs-algorithm brute_force
```
