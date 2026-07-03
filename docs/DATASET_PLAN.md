# Dataset plan

The full AGUVIS image collection is intentionally not part of the local MVP.
The development machine has limited free storage, and the complete public
collection is hundreds of gigabytes.

## Phase 1: checked-in fixture

- Eight synthetic trajectory records
- Eight-dimensional deterministic embeddings
- Exercises indexing, filtering, retrieval, and deduplication
- Requires only a few kilobytes

## Phase 2: public subsets (completed through 10K)

The current baseline imports 500 mobile-navigation rows through the Hugging
Face row API without downloading screenshots. The normalized JSON is about
1.4 MiB and references 3,557 images.

The storage-safe Parquet path now imports 10,000 mobile-navigation rows by
reading only `messages` and `metadata`; screenshot columns never leave the
remote shards. The local run contains 2,043 AITW and 7,957 Android Control
trajectories. It is embedded with 384-dimensional MiniLM for the CPU/GPU
benchmark and remains outside Git.

- Download only selected AGUVIS text columns.
- Store compressed screenshots outside Git.
- Cache embeddings separately from the source images.
- Use a content hash so repeated screenshots are embedded once.
- Keep a manifest containing source, license, split, and record ID.

## Phase 3: evaluation set (completed)

The checked-in human-reviewed evaluation set contains:

- 57 labeled AGUVIS trajectories in 10 retrieval-intent groups
- 44 duplicate-positive pairs and 197 negative or hard-negative pairs
- 57 controlled failure variants with exact-parent and alternative-success labels

It supports precision@k, recall@k, MRR, duplicate-pair F1, and parent-excluded
failure-recovery metrics. See `docs/EVALUATION.md` for the annotation policy,
results, and limitations.

## Phase 4: large-scale run

The 10K CPU/GPU run uses the NVIDIA Windows/WSL2 machine. The next scale points
are 100K and 1M vectors. The portable repository retains:

- Source code
- A small development sample
- Embeddings needed for local Lucene tests
- Aggregated benchmark results
