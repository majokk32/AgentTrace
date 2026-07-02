# Third-party data and model notice

AgentTrace source code is separate from the public data and model artifacts
used in its reproducible examples.

## AGUVIS sample

The JSON files under `sample-data/` contain normalized metadata and action text
from 500 rows of the public `cua-lite/Aguvis` `mobile.navigation` training
split. No screenshot bytes are included.

`cua-lite/Aguvis` is a preprocessed composite of
`xlangai/aguvis-stage1` and `xlangai/aguvis-stage2`. Its dataset card marks the
license as `other` and directs users to the original component datasets for
license and citation requirements:

- https://huggingface.co/datasets/cua-lite/Aguvis
- https://huggingface.co/datasets/xlangai/aguvis-stage1
- https://huggingface.co/datasets/xlangai/aguvis-stage2

Publishing this repository does not relicense those records.

## MiniLM

The optional embedding model is
`sentence-transformers/all-MiniLM-L6-v2`, licensed Apache-2.0:

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

The ONNX binary is intentionally excluded from Git. Download instructions,
pinned revision, and checksums are in `models/README.md`.

## Java dependencies

Maven resolves Apache Lucene, Jackson, JUnit, and Microsoft ONNX Runtime. Their
licenses remain with their respective projects and are not changed by this
repository.
