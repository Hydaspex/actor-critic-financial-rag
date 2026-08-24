from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class QdrantStore:
    """A document store implementation using Qdrant for vector similarity search."""
    def __init__(
        self,
        collection_name: str = "sec_filings",
        url: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
    ) -> None:





        qdrant_url = url or os.getenv("QDRANT_URL")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name

        model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedder = SentenceTransformer(model_name)

        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self) -> None:
        """Ensure that the Qdrant collection exists, creating it if necessary."""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if isinstance(exists, bool):
                if exists:
                    return
            else:
                if getattr(exists, "exists", False):
                    return
        except Exception:
            pass

        dim = self.embedder.get_sentence_embedding_dimension()
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
    def _ensure_payload_indexes(self) -> None:
        """Ensure that the necessary payload indexes exist for efficient filtering."""
        index_specs = {
            "ticker": models.PayloadSchemaType.KEYWORD,
            "cik": models.PayloadSchemaType.KEYWORD,
            "type": models.PayloadSchemaType.KEYWORD,
            "form_type": models.PayloadSchemaType.KEYWORD,
            "series_id": models.PayloadSchemaType.KEYWORD,
            "source_type": models.PayloadSchemaType.KEYWORD,
            "provider": models.PayloadSchemaType.KEYWORD,
            "filing_date": models.PayloadSchemaType.DATETIME,
        }

        for field_name, schema in index_specs.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                pass

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        """Ingest a list of documents into the Qdrant collection."""
        if not docs:
            return 0

        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        ids: list[str] = []

        for d in docs:
            text = d.get("text") or d.get("content")
            if not text:
                continue

            ids.append(str(d.get("id") or uuid.uuid4()))
            meta = d.get("metadata", {}) or {}
            payload = {"text": text, **meta}

            texts.append(text)
            payloads.append(payload)

        if not texts:
            return 0

        vectors = self.embedder.encode(texts, convert_to_numpy=True)

        points = [
            models.PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(texts))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 8,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the Qdrant collection by query, with an optional metadata filter."""
        if not query:
            return []

        query_vec = self.embedder.encode(query, convert_to_numpy=True)

        q_filter = None
        if metadata_filter:
            must = [
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v),
                )
                for k, v in metadata_filter.items()
            ]
            q_filter = models.Filter(must=must)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for h in getattr(hits, "points", []) or []:
            payload = getattr(h, "payload", None) or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            results.append(
                {
                    "text": text,
                    "metadata": meta,
                    "score": getattr(h, "score", None),
                }
            )
        return results

    def count(self) -> int:
        """Count the number of documents in the Qdrant collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
