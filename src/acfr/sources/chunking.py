from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup


def html_to_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    if "<html" in raw_text.lower() or "<body" in raw_text.lower():
        text = BeautifulSoup(raw_text, "html.parser").get_text("\n", strip=True)
    else:
        text = raw_text

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def make_chunk_records(
    text: str,
    metadata: dict[str, Any] | None = None,
    source_url: str | None = None,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    metadata = dict(metadata or {})
    clean_text = html_to_text(text)

    if not clean_text:
        return []

    if source_url:
        metadata["source_url"] = source_url

    chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)
    records: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        doc_id_src = (
            f"{metadata.get('ticker', '')}|{metadata.get('filing_date', '')}|"
            f"{source_url or ''}|{i}|{chunk[:80]}"
        )
        doc_id = hashlib.md5(doc_id_src.encode("utf-8")).hexdigest()

        records.append(
            {
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                },
            }
        )

    return records
