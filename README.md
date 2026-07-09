# Personal RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant that lets you talk to your documents — with multi-user accounts, conversation memory, guardrails, and a self-correcting (agentic) retrieval loop.

## Features

**Core RAG**
- Upload PDF, TXT, and Markdown files (20MB per file limit).
- Local vector search using ChromaDB and SentenceTransformers, with page-level metadata for PDFs.
- Hybrid retrieval: vector search re-ranked with a BM25 keyword score, so exact terms/acronyms aren't lost to pure semantic search.
- Answer generation via OpenRouter (cloud) or Ollama (fully local), with strict "answer only from context" prompting and source citations.
- Automatic de-duplication when re-uploading a file with the same name.

**Context & memory**
- Token-aware conversation history windowing (not a fixed message count).
- Follow-up questions are automatically reformulated into standalone search queries, with in-session caching to avoid redundant LLM calls.
- Long conversations get a rolling summary of older turns, so early context isn't lost once the active window scrolls past it.

**Agentic retrieval (Corrective RAG)**
- If the first retrieval attempt finds nothing, the pipeline automatically retries once with a widened search (more candidates, looser threshold) before giving up.
- The retry decision trace is visible in the UI's debug panel, so you can see when the traditional single-pass path was enough vs. when the agentic retry kicked in.

**Guardrails**
- Retrieved document content is treated as untrusted data, not instructions — with prompt-level defenses and logging against embedded prompt-injection attempts.
- An independent grounding check discards answers that aren't actually supported by the retrieved context.
- Per-user rate limiting (10 questions / 60 seconds) to prevent runaway API usage.
- Bcrypt password hashing (with automatic migration for older accounts).

**Multi-user accounts**
- Sign-up/login backed by Supabase, with per-user isolated document workspaces and chat history.

**Observability & evaluation**
- Every query is logged as structured JSON (query, retrieval quality, latency, agentic retry trace) to `data/logs/<username>.jsonl`.
- `evaluate_rag.py` — a standalone LLM-as-judge evaluation script that scores answer faithfulness and relevancy without needing a heavy eval framework.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   Create a `.env` file in the project root (or use Streamlit Secrets when deployed):
   ```env
   # LLM backend — required unless USE_OLLAMA=true
   OPENROUTER_API_KEY="your_openrouter_api_key"

   # Set to "true" to use a local Ollama model instead of OpenRouter
   USE_OLLAMA="false"

   # Multi-user accounts (Supabase). Without these, auth/signup is disabled.
   SUPABASE_URL="your_supabase_project_url"
   SUPABASE_KEY="your_supabase_api_key"

   # Optional — see "Advanced / opt-in features" below
   # EMBEDDING_MODEL="all-MiniLM-L6-v2"
   # ENABLE_RERANKER="false"
   ```
   If `USE_OLLAMA=true`, make sure [Ollama](https://ollama.com) is installed and running locally with the `llama3` model pulled.

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Sign up or log in (requires Supabase configuration).
2. Use the sidebar to upload one or multiple documents, then click **Index Documents**.
3. Ask questions in the main chat area — answers are grounded strictly in your uploaded documents.
4. Toggle **Show retrieved chunks** in Advanced Settings to see the retrieved chunks, their relevance distance, and the agentic retry trace (when triggered).
5. Adjust **Search Strictness** to control how lenient retrieval matching is.

## Advanced / opt-in features

These exist but are **off by default** to avoid changing behavior on an already-deployed index:

- `EMBEDDING_MODEL` — swap the SentenceTransformer embedding model. Changing this changes the vector space, so you must **Clear Indexed Data** and re-upload documents after switching.
- `ENABLE_RERANKER="true"` — adds a cross-encoder second-stage reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of hybrid search. Loads an extra model, so it adds memory/startup cost.

## Evaluating answer quality

`evaluate_rag.py` is a standalone dev tool (not part of the live app) that runs a set of test questions through the pipeline and scores each answer's faithfulness and relevancy using an LLM judge:

```bash
python evaluate_rag.py --username <your_username>
```

Edit the `TEST_CASES` list in the script to match documents you've actually indexed for that user. Results print to the console and save to `data/logs/eval_report_<username>.json`.

## Project structure

```
app.py                   Streamlit UI, auth, session/chat state
src/
  document_loader.py      Loads PDF/TXT/MD, page-level metadata for PDFs
  chunker.py               Splits documents into overlapping chunks
  vector_store.py          ChromaDB wrapper: hybrid search, dedup, clear
  generator.py              Answer generation, guardrails, LLM-as-judge
  rag_pipeline.py            Orchestrates retrieval + generation + memory
  context_utils.py            Token-aware conversation history helpers
  observability.py             Structured JSONL query logging
evaluate_rag.py            Standalone answer-quality evaluation script
```
