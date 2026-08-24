from __future__ import annotations

import json
from typing import Protocol

from acfr.rag.ollama_client import OllamaClient


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def _doc_id(ctx: dict, idx: int) -> str:
    """Resolve a stable identifier from a context dict returned by QdrantStore.search()."""
    meta = ctx.get("metadata") or {}
    return (
        meta.get("doc_id")
        or meta.get("id")
        or meta.get("ticker")
        or f"doc-{idx + 1}"
    )


class Critic:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or OllamaClient("CRITIC_MODEL")

    def build_prompt(self, query: str, contexts: list, answer: str, citations: list) -> str:
        evidence = "\n\n".join(
            [f"[{_doc_id(c, i)}] {c['text']}" for i, c in enumerate(contexts)]
        )
        citations_str = ", ".join(citations) if citations else "none"

        return (
            "You are a strict financial QA critic for a "
            "Retrieval-Augmented Generation (RAG) system. "
            "You will be given:\n"
            "- A user question\n"
            "- Evidence passages, each prefixed with a [doc_id]\n"
            "- A model answer and its list of cited doc_ids\n\n"
            "Your job is to evaluate the answer ONLY using the evidence provided. "
            "Ignore any outside knowledge.\n\n"
            "Evaluate on these dimensions (0.0-1.0 each):\n"
            "1) faithfulness_score: Are all factual claims in the answer "
            "supported by at least one evidence passage? "
            "Penalize hallucinations, contradictions, or claims not grounded "
            "in any [doc_id].\n"
            "2) completeness_score: Does the answer cover the key points in the "
            "evidence that are relevant to the question? "
            "Penalize missing major facts that the question implies should be "
            "addressed.\n"
            "3) citation_score: For each claim with a cited doc_id, does that "
            "document actually support the claim? "
            "Penalize incorrect or missing citations.\n\n"
            "Compute overall_score as the simple average of the three sub-scores.\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "overall_score": float,\n'
            '  "faithfulness_score": float,\n'
            '  "completeness_score": float,\n'
            '  "citation_score": float,\n'
            '  "issues": [string, ...]\n'
            "}\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Cited doc_ids: {citations_str}"
        )

    def _parse_actor_raw(self, draft_raw: str) -> tuple[str, list]:
        """
        Parse the actor's raw string as JSON {answer, citations}.
        Falls back to treating the whole string as plain-text answer.
        """
        try:
            data = json.loads(draft_raw)
            if isinstance(data, dict) and "answer" in data:
                answer = str(data.get("answer", ""))
                citations = data.get("citations", [])
                if not isinstance(citations, list):
                    citations = []
                return answer, [str(c) for c in citations]
        except Exception:
            pass
        return draft_raw, []

    def critique(self, query: str, contexts: list, draft_raw: str) -> dict:
        if not contexts:
            return {"score": 0.0, "notes": "No evidence available."}

        answer, citations = self._parse_actor_raw(draft_raw)
        prompt = self.build_prompt(query, contexts, answer, citations)
        text = self.client.generate(prompt)

        try:
            data = json.loads(text.strip())
            overall = float(data.get("overall_score", 0.0))
            faithfulness = float(data.get("faithfulness_score", 0.0))
            completeness = float(data.get("completeness_score", 0.0))
            citation_score = float(data.get("citation_score", 0.0))
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]

            notes_parts = [
                f"faithfulness={faithfulness:.2f}",
                f"completeness={completeness:.2f}",
                f"citations={citation_score:.2f}",
            ]
            if issues:
                notes_parts.append("issues: " + " | ".join(str(i) for i in issues))

            return {
                "score": overall,
                "notes": "; ".join(notes_parts),
            }
        except Exception:
            return {"score": 0.0, "notes": "Critic failed to parse JSON."}
