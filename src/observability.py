"""Structured JSONL query logging for observability.

Separate from Python's print()-based debug statements scattered elsewhere in
this codebase: those are for a human watching the Streamlit Cloud log stream
live. This module writes one JSON object per line to a per-user file, so past
activity (queries, retrieval quality, agentic retries, latency) can be
reviewed or analyzed later without needing to have been watching logs in
real time.
"""

import os
import json
import time
import logging

LOG_DIR = "data/logs"


def get_query_logger(username: str) -> logging.Logger:
    """Returns a logger that appends one JSON line per query to
    data/logs/<username>.jsonl. Safe to call repeatedly — reuses the same
    logger/handler instead of attaching duplicate handlers.
    """
    logger_name = f"rag_query_log.{username}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"{username}.jsonl"), encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_query_event(logger: logging.Logger, username: str, query: str, event: str, elapsed_ms: float, **extra):
    """Writes one structured JSON line describing how a query was handled.

    event: "casual" | "rate_limited" | "answered"
    extra: additional fields specific to the event (chunks_found, best_distance,
    retrieval_trace, fallback, etc.)
    """
    entry = {
        "timestamp": time.time(),
        "username": username,
        "query": query,
        "event": event,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    entry.update(extra)
    logger.info(json.dumps(entry, ensure_ascii=False))
