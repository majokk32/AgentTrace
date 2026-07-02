# Local embedding model

AgentTrace uses the official
`sentence-transformers/all-MiniLM-L6-v2` ONNX export pinned to revision:

```text
dfa9feb5cece5be2cc8fc23a3cf1f32473a9d56f
```

The model is Apache-2.0 licensed. Large ONNX files are intentionally ignored by
Git and are not included in the source archive.

## Apple Silicon

Download these two files:

```bash
mkdir -p models/all-MiniLM-L6-v2
curl -L --fail \
  'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/dfa9feb5cece5be2cc8fc23a3cf1f32473a9d56f/onnx/model_qint8_arm64.onnx?download=true' \
  -o models/all-MiniLM-L6-v2/model_qint8_arm64.onnx
curl -L --fail \
  'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/dfa9feb5cece5be2cc8fc23a3cf1f32473a9d56f/vocab.txt?download=true' \
  -o models/all-MiniLM-L6-v2/vocab.txt
```

Verify:

```text
4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474  model_qint8_arm64.onnx
07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  vocab.txt
```

File sizes are approximately 23 MB and 232 KB.

## Windows x64

The same pinned model repository provides
`onnx/model_quint8_avx2.onnx` for x64 CPUs. Use that file through the same
`--model` CLI option. ONNX Runtime's CUDA package can later be added separately;
the current semantic baseline deliberately uses CPU inference so vector-search
backend measurements remain isolated.
