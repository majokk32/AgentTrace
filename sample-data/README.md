# Sample data

## `trajectories.json`

Eight synthetic records used by unit and API smoke tests.

## `aguvis-500.json`

A normalized metadata-only sample of 500 public mobile-navigation trajectories
fetched from the Hugging Face row API for `cua-lite/Aguvis`.

- Source collection: `xlangai/aguvis-stage2`
- Normalized dataset: `cua-lite/Aguvis`
- Configuration: `mobile.navigation`
- Split: `train`
- Starting offset: `0`
- Imported rows: `500`
- Images downloaded: `0`
- Referenced screenshots: `3,557`
- Local embedding: 256-dimensional feature-hashing baseline generated from
  instructions and action descriptions

The feature-hashing vectors validate the importer and Lucene pipeline without
model weights.

## `aguvis-500-minilm.json`

The same 500 records re-embedded locally with the pinned
`sentence-transformers/all-MiniLM-L6-v2` ARM64 int8 ONNX model.

- Embedding dimension: `384`
- Input: instruction, app, and action descriptions
- Pooling: attention-mask-aware mean pooling followed by L2 normalization
- Inference runtime: ONNX Runtime Java CPU
- Local run: 500 records in 14.7 seconds (34.1 trajectories/second)

This is a real semantic text/action embedding, but it does not use screenshot
pixels. Retrieval quality is measured with the small human-reviewed label set
documented in `docs/EVALUATION.md`.

## `aguvis-500-plus-57-labeled-failures.json`

The 500 MiniLM records plus 57 controlled failure variants generated from the
human-reviewed retrieval label set. Each failure:

- is marked `success: false`
- retains the original task instruction
- keeps only the first two thirds of actions
- records the exact parent in `sourceId`
- records alternative same-intent successes in
  `labels/aguvis-labeled-failure-pairs-57.json`

These are controlled fault-injection records, not naturally collected failures.

See the original dataset card for licenses and citations:
https://huggingface.co/datasets/cua-lite/Aguvis
