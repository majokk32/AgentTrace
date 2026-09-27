# Operations and telemetry

AgentTrace includes a dependency-free operational dashboard for diagnosing the
running HTTP service. Start the service normally, then open:

- `http://localhost:8080/dashboard` for the live operator view
- `http://localhost:8080/api/telemetry` for the machine-readable snapshot
- `http://localhost:8080/health` for a lightweight liveness probe

## What is measured

The telemetry snapshot reports:

- request totals, in-flight work, successes, and 4xx/5xx counts;
- mean and maximum end-to-end HTTP latency;
- per-route request, error, and latency breakdowns;
- backend type, indexed trajectory count, and vector dimension;
- the latest server-side exception and diagnostic message; and
- explicit alerts when the server-error rate reaches 5% after at least 20
  workload requests, or when any workload request exceeds 1,000 ms.

`/health`, `/api/telemetry`, and `/dashboard` are deliberately excluded from
workload measurements. This prevents frequent probes and dashboard refreshes
from masking the behavior of search and deduplication traffic.

Metrics are process-local and reset on restart. They are intended for local
bring-up, validation, demos, and fast diagnosis—not as a replacement for a
durable production metrics pipeline.

## Triage workflow

1. Check `/health` to confirm the Java process is responsive.
2. Open `/dashboard` and look for error-rate or slow-request alerts.
3. Use the route table to identify the failing or slow operation.
4. Inspect `lastServerError` in `/api/telemetry` for the latest server-side
   failure type and message.
5. Compare the reported backend and index dimensions with the intended launch
   configuration before reproducing the request.

For the cuVS backend, also query the worker at `http://127.0.0.1:8765/health`
and confirm that its algorithm, index size, and dimensions match the Java
service.
