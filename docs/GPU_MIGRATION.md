# GPU migration plan

The Java/Lucene backend is the portable baseline. The NVIDIA backend will be
added behind the same `TrajectorySearchBackend` interface.

## Target machine

- Windows 11 host
- NVIDIA RTX 4070 Super
- Current NVIDIA driver
- WSL2 with Ubuntu for cuVS
- JDK 21 for the Java service

cuVS packages are Linux-oriented, so WSL2 is the clean boundary on a Windows
computer. Do not move the full dataset back to the storage-constrained Mac.

## Recommended split

```text
Windows / WSL2
  Java service
      |
      +-- LuceneTrajectorySearchBackend (CPU baseline)
      |
      +-- CuvsTrajectorySearchBackend (GPU batch index/search)
```

If direct Java/native integration blocks early progress, use a temporary
Python cuVS worker over localhost HTTP or gRPC. Keep the request contract
backend-neutral so it can later be replaced by direct Java Panama FFM calls.

## GPU milestone

1. Confirm the driver and GPU inside WSL2 with `nvidia-smi`.
2. Install a cuVS version compatible with the installed CUDA driver.
3. Load already-generated embeddings; do not run image models yet.
4. Implement GPU build/search with the same IDs and cosine semantics.
5. Verify returned neighbors against exact cosine ground truth.
6. Compare CPU and GPU only at matched recall.
7. Record index build time, p50/p95 latency, throughput, and memory.

## Portability rules

- Use `Path` rather than hard-coded path separators.
- Keep dataset locations in command-line arguments.
- Store benchmark results as JSON/CSV.
- Never commit generated indexes, images, model weights, or credentials.

