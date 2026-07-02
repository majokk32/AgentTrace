# Dataset plan

The full AGUVIS image collection is intentionally not part of the local MVP.
The development machine has limited free storage, and the complete public
collection is hundreds of gigabytes.

## Phase 1: checked-in fixture

- Eight synthetic trajectory records
- Eight-dimensional deterministic embeddings
- Exercises indexing, filtering, retrieval, and deduplication
- Requires only a few kilobytes

## Phase 2: small public subset (completed baseline)

The current baseline imports 500 mobile-navigation rows through the Hugging
Face row API without downloading screenshots. The normalized JSON is about
1.4 MiB and references 3,557 images.

Next target: 1,000 to 10,000 mobile navigation rows:

- Download only selected AGUVIS shards.
- Store compressed screenshots outside Git.
- Cache embeddings separately from the source images.
- Use a content hash so repeated screenshots are embedded once.
- Keep a manifest containing source, license, split, and record ID.

## Phase 3: evaluation set

Create a small human-reviewed file with:

- Query trajectory ID
- Relevant trajectory IDs
- Duplicate / not-duplicate labels
- Failure category

This enables precision@k, recall@k, and duplicate-pair F1. Those metrics are
more meaningful than reporting latency alone.

## Phase 4: large-scale run

Move images and GPU indexing to the NVIDIA machine. The Mac retains:

- Source code
- A small development sample
- Embeddings needed for local Lucene tests
- Aggregated benchmark results
