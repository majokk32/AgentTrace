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

AgentTrace source code is licensed under Apache-2.0. Third-party data and model
terms are documented separately in `THIRD_PARTY_NOTICES.md`.

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
  -jar target\agenttrace-0.2.0.jar `
  evaluate `
  --data sample-data\aguvis-500-minilm.json `
  --labels labels\aguvis-500-intents-v1.json `
  --index data\eval-minilm-index `
  --embedding-model all-MiniLM-L6-v2 `
  --k 5 `
  --duplicate-threshold 0.92 `
  --output reports\windows-minilm-evaluation.json
```

For the cuVS backend, use the pinned WSL2 environment and localhost worker:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/setup-wsl.sh
wsl.exe -d Ubuntu-24.04 -u root -- bash gpu/run-worker-wsl.sh
```

Then pass `--backend cuvs --cuvs-url http://127.0.0.1:8765` to the server or
evaluation commands. Do not commit generated environments, indexes, or model
weights.
