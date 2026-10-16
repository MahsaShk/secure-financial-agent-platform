from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import DocumentChunk


def chunks_from_records(records: list[dict[str, Any]]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        try:
            chunk_id = str(record["chunk_id"]).strip()
            source_id = str(record["source_id"]).strip()
            text = str(record["text"]).strip()
        except KeyError as error:
            raise ValueError(f"record {index} is missing {error.args[0]}") from error
        if not chunk_id or not source_id or not text:
            raise ValueError(f"record {index} has an empty required field")
        if chunk_id in seen_ids:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)
        chunks.append(DocumentChunk(chunk_id=chunk_id, source_id=source_id, text=text))
    if not chunks:
        raise ValueError("at least one document chunk is required")
    return chunks


def load_chunks(path: str | Path) -> list[DocumentChunk]:
    """Read a JSON list of document records from path and validate them into chunks."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("document file must contain a JSON list of objects")
    return chunks_from_records(payload)

