#!/usr/bin/env python3
"""Local cuVS worker for AgentTrace.

The worker binds to localhost by default and keeps trajectory metadata on the
host while cuVS stores searchable vectors on the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from cuvs.neighbors import brute_force
from pylibraft.common import device_ndarray


BACKEND_NAME = "cuvs-brute-force-gpu"


@dataclass
class CachedIndex:
    global_indices: np.ndarray
    vectors: device_ndarray
    index: Any


class AgentTraceGpuIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trajectories: list[dict[str, Any]] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._norms = np.empty((0,), dtype=np.float32)
        self._indices: dict[tuple[Any, ...], CachedIndex] = {}

    def rebuild(self, payload: Any) -> dict[str, Any]:
        trajectories = payload.get("trajectories") if isinstance(payload, dict) else payload
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError("at least one trajectory is required")

        ids: set[str] = set()
        dimension: int | None = None
        normalized: list[dict[str, Any]] = []
        embedding_rows: list[list[float]] = []
        for row in trajectories:
            if not isinstance(row, dict):
                raise ValueError("each trajectory must be a JSON object")
            trajectory_id = row.get("id")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError("each trajectory requires a non-empty id")
            if trajectory_id in ids:
                raise ValueError(f"duplicate trajectory id: {trajectory_id}")
            ids.add(trajectory_id)

            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"trajectory {trajectory_id} requires an embedding")
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError(f"all embeddings must have dimension {dimension}")
            embedding_rows.append(embedding)

            normalized.append(
                {
                    "id": trajectory_id,
                    "instruction": str(row.get("instruction", "")),
                    "platform": str(row.get("platform", "")),
                    "app": str(row.get("app", "")),
                    "success": bool(row.get("success", False)),
                    "actions": list(row.get("actions") or []),
                    "source": str(row.get("source") or ""),
                    "sourceId": str(row.get("sourceId") or ""),
                    "imageCount": int(row.get("imageCount", 0)),
                }
            )

        embeddings = np.ascontiguousarray(embedding_rows, dtype=np.float32)
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError("embeddings must not contain zero vectors")

        with self._lock:
            self._trajectories = normalized
            self._embeddings = embeddings
            self._norms = norms
            self._indices.clear()
            self._index_for((None, None, None))
            return self.stats()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            dimension = int(self._embeddings.shape[1]) if self._embeddings.size else 0
            return {
                "trajectoryCount": len(self._trajectories),
                "vectorDimension": dimension,
                "backend": BACKEND_NAME,
            }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": BACKEND_NAME,
            "cuvsVersion": version("cuvs-cu12"),
            "stats": self.stats(),
        }

    def search(self, request: Any) -> list[dict[str, Any]]:
        if not isinstance(request, dict):
            raise ValueError("search request must be a JSON object")
        with self._lock:
            if not self._trajectories:
                return []
            embedding = self._validate_query_embedding(request.get("embedding"))
            k = self._requested_k(request.get("k"))
            key = self._filter_key(request)
            cached = self._index_for(key)
            if cached.global_indices.size == 0:
                return []
            actual_k = min(k, int(cached.global_indices.size))
            distances, neighbors = self._search_index(
                cached, np.ascontiguousarray([embedding], dtype=np.float32), actual_k
            )

            results: list[dict[str, Any]] = []
            for distance, local_index in zip(distances[0], neighbors[0]):
                global_index = int(cached.global_indices[int(local_index)])
                result = dict(self._trajectories[global_index])
                result["score"] = float(np.clip(1.0 - float(distance) / 2.0, 0.0, 1.0))
                results.append(result)
            return results

    def deduplicate(self, request: Any) -> list[dict[str, Any]]:
        if not isinstance(request, dict):
            raise ValueError("deduplication request must be a JSON object")
        threshold = float(request.get("threshold", 0.985))
        candidate_k = int(request.get("candidateK", 10))
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be between -1 and 1")
        if candidate_k < 2 or candidate_k > 100:
            raise ValueError("candidateK must be between 2 and 100")

        with self._lock:
            if not self._trajectories:
                return []
            parent = list(range(len(self._trajectories)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(left: int, right: int) -> None:
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parent[max(left_root, right_root)] = min(left_root, right_root)

            metadata_groups: dict[tuple[str, str, bool], list[int]] = {}
            for index, trajectory in enumerate(self._trajectories):
                group_key = (
                    trajectory["platform"],
                    trajectory["app"],
                    trajectory["success"],
                )
                metadata_groups.setdefault(group_key, []).append(index)

            for group_key, global_indices in metadata_groups.items():
                key = (group_key[0], group_key[1], group_key[2])
                cached = self._index_for(key)
                actual_k = min(candidate_k, len(global_indices))
                queries = self._embeddings[cached.global_indices]
                _, neighbors = self._search_index(cached, queries, actual_k)
                for local_query, candidates in enumerate(neighbors):
                    left = int(cached.global_indices[local_query])
                    for local_candidate in candidates:
                        right = int(cached.global_indices[int(local_candidate)])
                        if left == right:
                            continue
                        if self._cosine(left, right) >= threshold:
                            union(left, right)

            groups: dict[int, list[int]] = {}
            for index in range(len(self._trajectories)):
                groups.setdefault(find(index), []).append(index)

            output: list[dict[str, Any]] = []
            for members in groups.values():
                if len(members) < 2:
                    continue
                members.sort(key=lambda index: self._trajectories[index]["id"])
                similarities = [
                    self._cosine(members[left], members[right])
                    for left in range(len(members))
                    for right in range(left + 1, len(members))
                ]
                member_ids = [self._trajectories[index]["id"] for index in members]
                output.append(
                    {
                        "canonicalId": member_ids[0],
                        "memberIds": member_ids,
                        "meanCosineSimilarity": float(
                            sum(similarities) / len(similarities)
                        ),
                    }
                )
            output.sort(key=lambda group: group["canonicalId"])
            return output

    def _validate_query_embedding(self, value: Any) -> np.ndarray:
        if not isinstance(value, list):
            raise ValueError("embedding is required")
        dimension = self._embeddings.shape[1]
        if len(value) != dimension:
            raise ValueError(f"embedding dimension must be {dimension}")
        embedding = np.asarray(value, dtype=np.float32)
        if not np.isfinite(embedding).all():
            raise ValueError("embedding must contain only finite values")
        if np.linalg.norm(embedding) == 0:
            raise ValueError("embedding must not be a zero vector")
        return embedding

    @staticmethod
    def _requested_k(value: Any) -> int:
        k = 5 if value is None else int(value)
        if k < 1 or k > 100:
            raise ValueError("k must be between 1 and 100")
        return k

    @staticmethod
    def _filter_key(request: dict[str, Any]) -> tuple[Any, ...]:
        platform = request.get("platform")
        app = request.get("app")
        success = request.get("success")
        platform = platform if isinstance(platform, str) and platform.strip() else None
        app = app if isinstance(app, str) and app.strip() else None
        if success is not None and not isinstance(success, bool):
            raise ValueError("success must be true, false, or null")
        return platform, app, success

    def _index_for(self, key: tuple[Any, ...]) -> CachedIndex:
        cached = self._indices.get(key)
        if cached is not None:
            return cached

        platform, app, success = key
        matching: list[int] = []
        for index, trajectory in enumerate(self._trajectories):
            if platform is not None and trajectory["platform"] != platform:
                continue
            if app is not None and trajectory["app"] != app:
                continue
            if success is not None and trajectory["success"] is not success:
                continue
            matching.append(index)

        global_indices = np.asarray(matching, dtype=np.int64)
        if not matching:
            cached = CachedIndex(global_indices, None, None)  # type: ignore[arg-type]
        else:
            host_vectors = np.ascontiguousarray(
                self._embeddings[global_indices], dtype=np.float32
            )
            gpu_vectors = device_ndarray(host_vectors)
            index = brute_force.build(gpu_vectors, metric="cosine")
            cached = CachedIndex(global_indices, gpu_vectors, index)
        self._indices[key] = cached
        return cached

    @staticmethod
    def _search_index(
        cached: CachedIndex, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        gpu_queries = device_ndarray(np.ascontiguousarray(queries, dtype=np.float32))
        distances, neighbors = brute_force.search(cached.index, gpu_queries, k)
        return distances.copy_to_host(), neighbors.copy_to_host()

    def _cosine(self, left: int, right: int) -> float:
        numerator = float(np.dot(self._embeddings[left], self._embeddings[right]))
        denominator = float(self._norms[left] * self._norms[right])
        if denominator == 0 or not math.isfinite(denominator):
            return 0.0
        return numerator / denominator


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], index: AgentTraceGpuIndex):
        super().__init__(address, WorkerHandler)
        self.index = index


class WorkerHandler(BaseHTTPRequestHandler):
    server: WorkerServer

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._write_json(HTTPStatus.OK, self.server.index.health())
            elif self.path == "/stats":
                self._write_json(HTTPStatus.OK, self.server.index.stats())
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exception:  # pragma: no cover - defensive HTTP boundary
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal server error", "detail": str(exception)},
            )

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/rebuild":
                result = self.server.index.rebuild(payload)
            elif self.path == "/search":
                result = self.server.index.search(payload)
            elif self.path == "/deduplicate":
                result = self.server.index.deduplicate(payload)
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._write_json(HTTPStatus.OK, result)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exception)})
        except Exception as exception:  # pragma: no cover - defensive HTTP boundary
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal server error", "detail": str(exception)},
            )

    def _read_json(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("request body is required")
        return json.loads(self.rfile.read(content_length))

    def _write_json(self, status: HTTPStatus, body: Any) -> None:
        payload = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTrace local cuVS worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = AgentTraceGpuIndex()
    if args.data is not None:
        with args.data.open(encoding="utf-8") as input_file:
            index.rebuild(json.load(input_file))
    server = WorkerServer((args.host, args.port), index)
    print(
        f"AgentTrace cuVS worker listening on http://{args.host}:{args.port} "
        f"with {index.stats()['trajectoryCount']} trajectories",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
