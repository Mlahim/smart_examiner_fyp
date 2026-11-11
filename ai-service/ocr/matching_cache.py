"""
Smart Examiner — Matching Results Cache (SQLite)

Caches normalized LLM matching results to guarantee reproducibility and save API calls.
Cache keys are deterministic fingerprints of the inputs.
"""

import sqlite3
import json
import logging
from typing import Any, Optional
import hashlib
import os

from config import MATCHING_MODEL_NAME, MATCHING_PROMPT_VERSION

logger = logging.getLogger(__name__)

# Keep cache versioned. Bumping this invalidates all past matching caches.
MATCHING_CACHE_VERSION = "3.0"

def _get_db_path() -> str:
    db_path = os.environ.get("MATCHING_CACHE_DB", "matching_cache.sqlite")
    return db_path

def _init_db():
    path = _get_db_path()
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS matching_cache (
                fingerprint TEXT PRIMARY KEY,
                normalized_result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

# Ensure table exists
_init_db()

def _compute_fingerprint(
    student_norm_id: str,
    student_answer: str,
    blueprint_hash: str,
    prompt_version: str = MATCHING_PROMPT_VERSION,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a matching execution."""
    components = [
        f"cache_version={MATCHING_CACHE_VERSION}",
        f"model={MATCHING_MODEL_NAME}",
        f"prompt_version={prompt_version}",
        f"blueprint_hash={blueprint_hash}",
        f"student_question_id={student_norm_id}",
        f"student_answer={student_answer.strip()}",
    ]
    raw_str = "|".join(components)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

def get_cached_matching(
    student_norm_id: str,
    student_answer: str,
    blueprint_hash: str,
) -> Optional[dict[str, Any]]:
    """Retrieve a cached normalized matching result if it exists."""
    fingerprint = _compute_fingerprint(student_norm_id, student_answer, blueprint_hash)
    try:
        with sqlite3.connect(_get_db_path()) as conn:
            cursor = conn.execute(
                "SELECT normalized_result FROM matching_cache WHERE fingerprint = ?",
                (fingerprint,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
    except Exception as e:
        logger.warning(f"Failed to read from matching cache: {e}")
    return None

def set_cached_matching(
    student_norm_id: str,
    student_answer: str,
    blueprint_hash: str,
    normalized_result: dict[str, Any],
) -> None:
    """Store a normalized matching result in the cache."""
    fingerprint = _compute_fingerprint(student_norm_id, student_answer, blueprint_hash)
    try:
        with sqlite3.connect(_get_db_path()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO matching_cache (fingerprint, normalized_result) VALUES (?, ?)",
                (fingerprint, json.dumps(normalized_result))
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to write to matching cache: {e}")
