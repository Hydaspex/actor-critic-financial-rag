from __future__ import annotations

import os
from typing import Any, Protocol

import acfr.rag.store_qdrant as store_qdrant
from acfr.rag.actor import Actor
from acfr.rag.critic import Critic


class DocumentStore(Protocol):
    def ingest(self, docs: list[dict[str, Any]]) -> int: ...

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...


class RagController:
    """Thin orchestrator: store retrieves, actor generates, critic scores."""

    def __init__(self, store: DocumentStore | None = None) -> None:
        self.store = store or store_qdrant.QdrantStore()
        self.actor = Actor()
        self.critic = Critic()
        self.min_score = float(os.getenv("CRITIC_MIN_SCORE", "0.8"))

    # ------------------------------------------------------------------
    # Raw store operations
    # ------------------------------------------------------------------

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        return self.store.ingest(docs)

    def query(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return raw retrieval hits only (no LLM synthesis)."""
        hits = self.store.search(query=query, top_k=top_k, metadata_filter=metadata_filter)
        return {"query": query, "top_k": top_k, "results": hits}

    def count(self) -> int:
        return self.store.count()

    # ------------------------------------------------------------------
    # Full actor-critic pipeline (orchestration lives here, NOT in API)
    # ------------------------------------------------------------------

    def answer_with_critique(
        self,
        query: str,
        top_k: int = 8,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve -> Actor generates answer -> Critic scores -> return enriched payload.

        Returns
        -------
        {
            "query":        str,
            "status":       "accepted" | "abstained",
            "answer":       str,
            "citations":    list[str],
            "critic_score": float,
            "critic_notes": str,
            "results":      list[dict],   # raw retrieved docs for transparency
        }
        """
        contexts = self.store.search(
            query=query, top_k=top_k, metadata_filter=metadata_filter
        )

        actor_out = self.actor.act(query, contexts)
        critic_out = self.critic.critique(query, contexts, actor_out["raw"])

        accepted = critic_out["score"] >= self.min_score
        return {
            "query": query,
            "status": "accepted" if accepted else "abstained",
            "answer": actor_out["answer"],
            "citations": actor_out["citations"],
            "critic_score": critic_out["score"],
            "critic_notes": critic_out["notes"],
            "results": contexts,
        }
