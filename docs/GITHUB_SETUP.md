# GitHub setup and Windows handoff

## Publish from this Mac

Create an empty GitHub repository named `AgentTrace`. Do not initialize it with
a README, license, or `.gitignore`, because those files already exist locally.

From the `agenttrace` directory:

```bash
git status
git add .
git commit -m "Build AgentTrace CPU vector-search evaluation pipeline"
git remote add origin https://github.com/YOUR_USERNAME/AgentTrace.git
git push -u origin main
```

The repository intentionally excludes:

- `target/` build output
- `data/` Lucene indexes
- `work/` portable JDK and tools
- `models/**/*.onnx` model binaries
- IDE and macOS metadata

Before making the repository public, choose and add a source-code license if
you want others to have permission to reuse the implementation. Third-party
data/model terms are documented separately in `THIRD_PARTY_NOTICES.md`.

## Clone on the RTX 4070 Super computer

Native Windows CPU verification:

```powershell
git clone https://github.com/YOUR_USERNAME/AgentTrace.git
cd AgentTrace
.\mvnw.cmd clean package
```

The committed JSON already contains MiniLM vectors, so running the Lucene
evaluation does not require downloading the model:

```powershell
java --add-modules jdk.incubator.vector `
  --enable-native-access=ALL-UNNAMED `
  -jar target\agenttrace-0.1.0-SNAPSHOT.jar `
  evaluate `
  --data sample-data\aguvis-500-minilm.json `
  --labels labels\aguvis-500-intents-v1.json `
  --index data\eval-minilm-index `
  --embedding-model all-MiniLM-L6-v2 `
  --k 5 `
  --duplicate-threshold 0.92 `
  --output reports\windows-minilm-evaluation.json
```

For the cuVS stage, use WSL2/Linux with NVIDIA GPU access. First verify that
`nvidia-smi` works inside WSL2; then pin CUDA/cuVS versions before adding the
new backend. Do not commit generated indexes or CUDA build output.
