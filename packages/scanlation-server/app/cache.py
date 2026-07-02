"""SQLite result cache + manual translation memory (TM).

Two concerns:
  * ocr_runs  — a whole page's result keyed by (md5, langs, engines, opt_hash).
                Powers the lazy flow: client POSTs md5 only; a hit returns the
                stored result without re-running models.
  * translations — every (src_text, langs, model)->dst_text. model='manual'
                   wins (favor_manual) so user corrections persist.

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
CREATE TABLE IF NOT EXISTS translations (
    src_text TEXT, src_lang TEXT, dst_lang TEXT, model TEXT,
    dst_text TEXT NOT NULL,
    created_at REAL DEFAULT (strftime('%s','now')),
    PRIMARY KEY (src_text, src_lang, dst_lang, model)
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

    # --- page result cache (lazy flow) ---
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

    def clear_runs(self) -> int:
        """Drop every cached page result so pages re-run the full pipeline on next
        view. Translation memory (``translations``, incl. manual corrections) is
        left untouched — machine translations are recomputed anyway, and manual
        edits are user data. Returns the number of cached pages removed."""
        with self._lock:
            n = self._conn.execute("DELETE FROM ocr_runs").rowcount
            self._conn.commit()
        return n

    # --- translation memory ---
    def get_translations(self, src_text: str, src_lang: str, dst_lang: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, dst_text FROM translations WHERE src_text=? AND src_lang=? AND dst_lang=? "
                "ORDER BY (model='manual') DESC, created_at DESC",
                (src_text, src_lang, dst_lang),
            ).fetchall()
        return [{"model": r[0], "text": r[1]} for r in rows]

    def best_translation(self, src_text: str, src_lang: str, dst_lang: str) -> Optional[dict]:
        rows = self.get_translations(src_text, src_lang, dst_lang)
        if not rows:
            return None
        if settings.favor_manual:
            for r in rows:
                if r["model"] == "manual":
                    return r
        return rows[0]

    def put_translation(self, src_text: str, src_lang: str, dst_lang: str, model: str, dst_text: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO translations(src_text, src_lang, dst_lang, model, dst_text) VALUES (?,?,?,?,?)",
                (src_text, src_lang, dst_lang, model, dst_text),
            )
            self._conn.commit()


cache = Cache()
