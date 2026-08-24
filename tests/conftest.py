"""Session-scoped stubs so tests never need live Qdrant or Ollama.

PATCHING STRATEGY
-----------------
autouse fixtures run per-test, AFTER all modules are imported.  But
`acfr.api.main` constructs RagController() at module level, which calls
OllamaClient() and QdrantStore() at *import time* - before any fixture
can run.

Fix: use a session-scoped autouse fixture that patches both classes at
the CLASS level on their source modules before the first test is
collected.  Per-test fixtures then remain as a safety net.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# _StubStore  (no-dep in-memory DocumentStore)
# ---------------------------------------------------------------------------

class _StubStore:
    def __init__(self, *args, **kwargs):
        self._docs: list = []

    def ingest(self, docs):
        self._docs.extend(docs)
        return len(docs)

    def search(self, query, top_k=8, metadata_filter=None):
        return [
            {"text": d["text"], "metadata": d.get("metadata", {}), "score": 1.0}
            for d in self._docs
            if query.lower() in d["text"].lower()
        ][:top_k]

    def count(self):
        return len(self._docs)


# ---------------------------------------------------------------------------
# _StubOllamaClient  (never touches localhost:11434)
# ---------------------------------------------------------------------------

_ACTOR_RESPONSE = '{"answer":"stub answer","citations":["doc-1"]}'
_CRITIC_RESPONSE = (
    '{"overall_score":0.9,"faithfulness_score":0.95,'
    '"completeness_score":0.88,"citation_score":0.87,"issues":[]}'
)

_CRITIC_MARKERS = (
    "faithfulness_score",
    "citation_score",
    "completeness_score",
    "strict financial QA critic",
)


class _StubOllamaClient:
    """Safe no-op LLM client.  __init__ accepts any kwargs without raising."""

    def __init__(self, *args, **kwargs):   # <-- accepts model_env_key or anything
        pass

    def generate(self, prompt: str) -> str:
        if any(marker in prompt for marker in _CRITIC_MARKERS):
            return _CRITIC_RESPONSE
        return _ACTOR_RESPONSE


# ---------------------------------------------------------------------------
# SESSION-SCOPED patch  — runs once, before any module is imported by tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _patch_services_session():
    """Patch QdrantStore and OllamaClient at class level for the whole session.

    This runs before any test module is imported, so module-level
    RagController() construction in acfr.api.main never sees the real classes.
    """
    import acfr.rag.actor as actor_mod
    import acfr.rag.controller as ctrl
    import acfr.rag.critic as critic_mod
    import acfr.rag.ollama_client as oc
    import acfr.rag.store_qdrant as sq

    # Patch at the source so every subsequent import gets the stub
    sq.QdrantStore         = _StubStore
    oc.OllamaClient        = _StubOllamaClient
    actor_mod.OllamaClient = _StubOllamaClient
    critic_mod.OllamaClient = _StubOllamaClient

    # Also fix the reference already held by the controller module
    ctrl.store_qdrant = type("_mod", (), {"QdrantStore": _StubStore})()

    yield  # tests run here


# ---------------------------------------------------------------------------
# PER-TEST monkeypatch fixtures  (safety net; also resets between tests)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_qdrant(monkeypatch):
    import acfr.rag.store_qdrant as sq
    monkeypatch.setattr(sq, "QdrantStore", _StubStore)
    import acfr.rag.controller as ctrl
    monkeypatch.setattr(ctrl, "store_qdrant", type(
        "_mod", (), {"QdrantStore": _StubStore}
    )())


@pytest.fixture(autouse=True)
def _stub_ollama(monkeypatch):
    import acfr.rag.ollama_client as oc
    monkeypatch.setattr(oc, "OllamaClient", _StubOllamaClient)
    import acfr.rag.actor as actor_mod
    monkeypatch.setattr(actor_mod, "OllamaClient", _StubOllamaClient)
    import acfr.rag.critic as critic_mod
    monkeypatch.setattr(critic_mod, "OllamaClient", _StubOllamaClient)
