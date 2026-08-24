# Actor–Critic Financial RAG

A portfolio‑grade **actor–critic Retrieval‑Augmented Generation (RAG)** system for financial research over SEC filings and macro data. Designed to run entirely on free‑tier / open‑source components and to look like a realistic PE / hedge‑fund research assistant rather than a toy demo.

> Use‑case: “Given recent filings and macro conditions, explain what changed in revenue, margins, liquidity, or capital allocation — with evidence and explicit risk controls.”

---

## Motivation

Recent financial QA benchmarks (e.g., FinanceBench and Fin‑RATE) show that even strong LLMs struggle with open‑book questions over SEC filings, especially when answers must track a company over time and remain tightly grounded in evidence. Performance drops sharply when you move from single‑document lookups to cross‑entity comparisons and longitudinal tracking, and many errors come from retrieval and hallucination rather than raw model capability.

This project is built to address those pain points:

- Use **real financial sources** (SEC + FRED), not synthetic text.
- Make retrieval a **first‑class concern**, not an afterthought.
- Wrap generation in an **actor–critic safety loop** that can veto low‑quality answers and encourage abstention instead of confident hallucinations.

It’s intended as a **portfolio‑ready repo** that demonstrates production thinking around financial RAG, evaluation, and risk control.

---

## Features

- **Actor–critic RAG loop**
  - **Actor**: drafts an investment answer grounded in retrieved SEC/FRED evidence with explicit document‑level citations.
  - **Critic**: scores the draft on a 0–1 scale for grounding, numeric fidelity, citation quality, and safety, returning structured feedback.
  - **Controller**: decides whether to **accept** or **abstain** based on critic score; abstain responses include critic notes for human review.

- **Hybrid retrieval (local, free‑tier)**
  - BM25 + TF‑IDF hybrid scoring over a local store of documents.
  - Tunable `top_k`, ready to be swapped to Chroma/FAISS/pgvector when scaling up.

- **Investment‑grade data sources**
  - **SEC EDGAR submissions** via `data.sec.gov` (10‑K, 10‑Q, etc.).
  - **FRED macro time series** for rate and macro context (e.g. Treasury yields).
  - Bundled sample corpus for out‑of‑the‑box demo.

- **Golden dataset + evaluation harness**
  - CSV‑based golden dataset with `query`, `gold_doc_ids`, `reference_answer`, `task_type`, `risk_level`, `notes`.
  - Offline evaluation: `recall@k`, `precision@k`, `MRR@k`, `NDCG@k`, `answer_match`, `critic_score` average.

- **Production‑style plumbing**
  - FastAPI backend with clear endpoints.
  - Dockerfile + `docker-compose.yml` for containerized deployment.
  - GitHub Actions CI with smoke tests and metric tests.
  - Scripts for sample ingestion and golden‑set curation.

Everything runs **locally** (Python + Ollama + free public APIs) with no paid SaaS dependencies.

---

## Architecture

High‑level flow:

1. **Ingestion**
   - Documents are loaded from SEC / FRED (or your own sources).
   - Each document is normalized into a canonical schema:
     - `doc_id`, `source`, `text`, `metadata` (ticker, filing type, series ID, etc.).
   - Stored in a local corpus for retrieval.

2. **Retrieval**
   - BM25 + TF‑IDF hybrid scoring: lexical relevance plus semantic similarity.
   - Top‑k evidence chunks returned for a query with scores.

3. **Actor**
   - Builds an evidence‑rich prompt with `[doc_id]` tags next to each passage.
   - Drafts a concise answer **only from the evidence**, instructed to cite `[doc_id]` next to claims and avoid guessing.

4. **Critic**
   - Receives the same evidence plus the actor’s draft.
   - Emits strict JSON:
     ```json
     { "score": float, "notes": string }
     ```
   - Evaluates:
     - Evidence grounding.
     - Numeric fidelity with the underlying SEC/FRED data.
     - Citation correctness (doc_ids match claims).
     - Safety / conservatism in phrasing.
   - Inspired by critique‑guided agent frameworks and actor–critic agent patterns where a dedicated critic guides and constrains the actor.

5. **Controller**
   - Compares `critic_score` against `CRITIC_MIN_SCORE` (configurable via `.env`).
   - Returns:
     - `status: "accepted"` for high‑score answers.
     - `status: "abstained"` for low‑score answers, plus critic notes and evidence for manual review.

This design explicitly models a **safety gate** for financial QA, consistent with recommendations to combine generation with hallucination checking in RAG systems.

---

## Alignment with financial QA benchmarks

This repo is not a direct implementation of any specific benchmark, but it aligns with the kind of questions and evaluation style used in:

- **FinanceBench** – open‑book QA over SEC filings with evidence strings and realistic analyst questions.
- **Fin‑RATE** – detail‑oriented reasoning, cross‑entity comparison, and longitudinal tracking over SEC filings.

You can treat this codebase as an “applied” counterpart to such benchmarks: it’s structured so you can plug in similar evaluation ideas and corpora, without hard‑coding any external dataset.

---

## Stack

- **Backend:** FastAPI + Uvicorn.
- **Models:** Local LLMs via Ollama (one model used as actor, one as critic; can be the same or different).
- **Retrieval:** BM25 + TF‑IDF hybrid over local docs (no external vector DB required).
- **Data:** SEC EDGAR submissions from `data.sec.gov` + FRED macro series.
- **Evaluation:** Pandas‑based harness over CSV golden dataset.
- **Ops:** Docker, docker‑compose, GitHub Actions CI, pytest.

---

## Quick start

### 1. Prerequisites

- Python 3.11
- Ollama running locally, with at least one pulled model:
  ```bash
  ollama pull llama3.1:8b
  ```
- Git, Docker (optional but recommended).

### 2. Setup

```bash
git clone https://github.com/<your-username>/actor-critic-financial-rag.git
cd actor-critic-financial-rag

cp .env.example .env
# Edit .env:
# - SEC_USER_AGENT="Your Name your.email@domain.com"
# - FRED_API_KEY="<optional>"
# - ACTOR_MODEL="llama3.1:8b"
# - CRITIC_MODEL="llama3.1:8b"
# - CRITIC_MIN_SCORE="0.8"  # suggested default

make install
make ingest-sample
make dev
```

The API will start at `http://localhost:8000`.

### 3. Health check

```bash
curl http://localhost:8000/v1/health
```

You should see:

```json
{ "status": "ok" }
```

---

## API overview

### `GET /v1/health`

Simple health check; returns `{ "status": "ok" }` if the server is up.

### `POST /v1/ingest`

Ingest your own documents.

- Body:
  ```json
  {
    "documents": [
      {
        "doc_id": "sec_10k_2025_apple",
        "source": "sec",
        "text": "Revenue increased ...",
        "metadata": { "ticker": "AAPL", "type": "10-K" }
      }
    ]
  }
  ```

- Response: number of documents indexed.

### `POST /v1/ingest-sample`

Load the bundled SEC/FRED‑like sample corpus. Good for quick demos.

- Response: number of documents indexed.

### `POST /v1/query` (actor–critic flow)

Run the full actor–critic RAG loop.

- Body:
  ```json
  {
    "query": "What changed in revenue and liquidity?",
    "top_k": 5
  }
  ```

- Response structure (simplified):
  ```json
  {
    "query": "What changed in revenue and liquidity?",
    "status": "accepted",
    "actor_answer": "... [sec_10k_2025_apple] ...",
    "citations": ["sec_10k_2025_apple", "fred_dgs10_2026"],
    "critic_score": 0.87,
    "critic_notes": "Grounded, numbers consistent with evidence.",
    "retrieved_docs": [...]
  }
  ```

If `critic_score < CRITIC_MIN_SCORE`, `status` will be `"abstained"` and the notes will explain why.

### `POST /v1/eval`

Offline evaluation harness against a golden dataset.

- Expects:
  - `predictions`: list of outputs from `/v1/query`.
  - `golden`: list of golden examples (query, gold_doc_ids, reference_answer, etc.).
- Returns per‑query metrics and an aggregate summary:
  - `recall@k`, `precision@k`, `MRR@k`, `NDCG@k`
  - `answer_match`
  - `critic_score` average

---

## Golden dataset & evaluation

### Golden seed

The repo includes a starter `data/golden_seed.csv` with rows like:

- `query`
- `gold_doc_ids` (list of doc_ids)
- `reference_answer`
- `task_type`
- `risk_level`
- `notes`

These are simple examples over the bundled SEC/FRED sample.

### Curate the golden set

Run:

```bash
python scripts/build_golden.py
```

This will produce `data/golden_curated.csv`, which is a cleaned version ready to feed into the eval harness.

### Running offline eval

You can collect predictions (e.g. by hitting `/v1/query` in batch) and then call the `evaluate` function in `src/eval/harness.py` to compute metrics:

- **Retrieval quality:** `recall@k`, `precision@k`, `MRR@k`, `NDCG@k`.
- **Answer quality:** `answer_match`, `critic_score` average.

This gives you a concrete way to detect regressions when changing:

- chunking strategy,
- retrieval scoring,
- actor/critic prompts or models,
- corpus composition.

---

## Actor–critic details

### Actor

- Builds a prompt that:
  - Frames the model as an investment research assistant.
  - Injects evidence chunks tagged with `[doc_id]`.
  - Instructs “answer only from evidence” and “always cite `[doc_id]` next to claims”.
- Output is parsed into:
  - `actor_answer` (final text),
  - `citations` (subset of doc_ids),
  - `raw` (unmodified model output for critic input).

### Critic

- Receives the same evidence and the actor’s draft.
- Asked to return strict JSON with:
  - `score`: float in `[0,1]`
  - `notes`: free‑text explanation
- Evaluates:
  - Evidence grounding.
  - Numeric fidelity (no invented revenue / margin / yield numbers).
  - Citation alignment (claims point to the right documents).

### Controller

- Compares `critic_score` with `CRITIC_MIN_SCORE`.
- Returns:
  - `status: "accepted"` when score is high enough.
  - `status: "abstained"` otherwise, with critic notes.

Currently the controller does not perform multi‑step revisions; it is designed as a **single‑pass gate** suitable for a risk‑aware financial QA assistant. Multi‑round critique and revision can be added as a natural extension.

---

## Roadmap

Potential extensions:

- Plug in **Chroma / FAISS / pgvector** for larger corpora.
- Add adapters to evaluate directly on public benchmarks like FinanceBench and Fin‑RATE.
- Upgrade critic to multi‑round critique + revision, aligning more closely with full critique‑guided frameworks.
- Add **risk/compliance checks** (e.g., detecting speculative language or missing risk caveats).
- Wire up a simple UI (Streamlit or React) for interactive research sessions.

---

## License

MIT – see `LICENSE` for details.