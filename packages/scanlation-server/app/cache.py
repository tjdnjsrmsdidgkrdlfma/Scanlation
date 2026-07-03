"""SQLite page-result cache.

  * ocr_runs — a whole page's result keyed by (md5, langs, engines, opt_hash).
               Powers the lookup flow: the client probes /run_lookup/ with md5
               only; a hit returns the stored result without re-running the models.

WAL + a process lock keep it safe under the threadpool that runs blocking work.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Optional

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    md5 TEXT PRIMARY KEY,
    created_at REAL DEFAULT (strftime('%s','now'))
);
CREATE TABLE IF NOT EXISTS ocr_runs (
    md5 TEXT, src TEXT, dst TEXT, engines TEXT, opt_hash TEXT,
    result_json TEXT NOT NULL,
    created_at REAL DEFAULT (strftime('%s','now')),
    PRIMARY KEY (md5, src, dst, engines, opt_hash)
);
"""


def opt_hash(*option_dicts: dict) -> str:
    """Stable sha256 over normalized option dicts."""
    norm = json.dumps(option_dicts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class Cache:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            settings.data_dir / "scanlation.sqlite", check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- page result cache (lookup flow) ---
    def get_run(self, md5: str, src: str, dst: str, engines: str, oh: str) -> Optional[list[dict]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM ocr_runs WHERE md5=? AND src=? AND dst=? AND engines=? AND opt_hash=?",
                (md5, src, dst, engines, oh),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_run(self, md5: str, src: str, dst: str, engines: str, oh: str, result: list[dict]) -> None:
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO images(md5) VALUES (?)", (md5,))
            self._conn.execute(
                "INSERT OR REPLACE INTO ocr_runs(md5, src, dst, engines, opt_hash, result_json) VALUES (?,?,?,?,?,?)",
                (md5, src, dst, engines, oh, json.dumps(result, ensure_ascii=False)),
            )
            self._conn.commit()

    def clear(self) -> int:
        """Drop all cached page results so everything is recomputed fresh on next
        use. Returns the rows removed."""
        with self._lock:
            n = self._conn.execute("DELETE FROM ocr_runs").rowcount
            self._conn.commit()
        return n


cache = Cache()
