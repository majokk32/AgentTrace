"""Fetch AGUVIS text rows without downloading screenshot columns.

Requires requests, fsspec, and PyArrow 19.0.1 or newer. PyArrow 19.0.0 has a
known nested-Parquet regression and must not be used for this dataset.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import fsspec
import pyarrow
import pyarrow.parquet as parquet
import requests


PARQUET_API = "https://datasets-server.huggingface.co/parquet"
DATASET = "cua-lite/Aguvis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch text-only AGUVIS rows through Parquet column pruning"
    )
    parser.add_argument("--config", default="mobile.navigation")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("work/aguvis-10000-rows.json")
    )
    return parser.parse_args()


def parquet_urls(config: str, split: str) -> list[str]:
    response = requests.get(
        PARQUET_API,
        params={"dataset": DATASET},
        headers={"User-Agent": "AgentTrace/0.2"},
        timeout=60,
    )
    response.raise_for_status()
    files = [
        item
        for item in response.json()["parquet_files"]
        if item["config"] == config and item["split"] == split
    ]
    files.sort(key=lambda item: item["filename"])
    if not files:
        raise RuntimeError(f"no Parquet files found for {config}/{split}")
    return [item["url"] for item in files]


def read_shard(url: str) -> list[tuple[str, str]]:
    last_failure: Exception | None = None
    for attempt in range(1, 4):
        try:
            with fsspec.open(url, "rb", block_size=1 << 20) as source:
                table = parquet.ParquetFile(source).read(
                    columns=["messages", "metadata"]
                )
            messages = table.column("messages").to_pylist()
            metadata = table.column("metadata").to_pylist()
            return list(zip(messages, metadata, strict=True))
        except Exception as failure:
            last_failure = failure
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"failed to read {url}") from last_failure


def image_count(messages_text: str) -> int:
    indices: set[int] = set()
    unindexed = 0
    for message in json.loads(messages_text):
        for item in message.get("content") or []:
            if item.get("type") != "image":
                continue
            index = item.get("index")
            if isinstance(index, int):
                indices.add(index)
            else:
                unindexed += 1
    return max(indices, default=-1) + 1 if indices else unindexed


def row_wrapper(
    row_index: int, messages: str, metadata: str
) -> dict[str, Any]:
    return {
        "row_idx": row_index,
        "row": {
            "messages": messages,
            "metadata": metadata,
            "images": [None] * image_count(messages),
        },
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be between 1 and 16")
    if pyarrow.__version__ == "19.0.0":
        raise RuntimeError(
            "PyArrow 19.0.0 cannot read these nested Parquet files; "
            "install 19.0.1 or newer"
        )

    urls = parquet_urls(args.config, args.split)
    rows: list[dict[str, Any]] = []
    next_progress = 1_000
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(urls), args.workers):
            shard_rows = executor.map(
                read_shard, urls[start : start + args.workers]
            )
            for shard in shard_rows:
                for messages, metadata in shard:
                    if len(rows) == args.limit:
                        break
                    rows.append(row_wrapper(len(rows), messages, metadata))
                if len(rows) == args.limit:
                    break
            if len(rows) >= next_progress:
                print(f"Fetched {len(rows):,} of {args.limit:,} text rows")
                next_progress = (len(rows) // 1_000 + 1) * 1_000
            if len(rows) == args.limit:
                break

    if len(rows) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} rows but found only {len(rows)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Wrote {len(rows):,} AGUVIS text rows to {args.output}")


if __name__ == "__main__":
    main()
