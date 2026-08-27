"""SQLite page-result cache.

  * page_runs — a whole page's pipeline result keyed by (md5, langs, engines,
                opt_hash). Powers the lookup flow: the client probes /run_lookup/
                with md5 only; a hit returns the stored result without re-running
                the models.

WAL + a process lock keep it safe under the threadpool that runs blocking work.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
import threading
from typing import Optional

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS page_runs (
    md5 TEXT, src TEXT, dst TEXT, engines TEXT, opt_hash TEXT,
    result_json TEXT NOT NULL,
    created_at REAL DEFAULT (strftime('%s','now')),
    PRIMARY KEY (md5, src, dst, engines, opt_hash)
);
CREATE TABLE IF NOT EXISTS page_stats (
    id INTEGER PRIMARY KEY,
    created_at REAL DEFAULT (strftime('%s','now')),
    engines TEXT, src TEXT, dst TEXT, md5 TEXT, skip_translate INTEGER DEFAULT 0,
    regions INTEGER, raw_regions INTEGER,
    decode_ms REAL, lockwait_ms REAL, detect_ms REAL, recognize_ms REAL,
    semwait_ms REAL, translate_ms REAL, total_ms REAL
);
CREATE TABLE IF NOT EXISTS region_stats (
    id INTEGER PRIMARY KEY,
    page_id INTEGER,
    crop_w INTEGER, crop_h INTEGER,
    score REAL, source_len INTEGER, dest_len INTEGER, recognize_ms REAL
);
"""

# Numeric columns the summary aggregates (min/max/mean/median/p90/p99). vertical /
# area / aspect are intentionally NOT stored — they derive from crop_w/crop_h.
_PAGE_COLS = ("regions", "raw_regions", "decode_ms", "lockwait_ms", "detect_ms",
              "recognize_ms", "semwait_ms", "translate_ms", "total_ms")
_REGION_COLS = ("crop_w", "crop_h", "score", "source_len", "dest_len", "recognize_ms")

# Display precision for a summary row. Every statistic in the row is rounded the same
# way, at the row's own scale — one fixed decimal count can't serve both milliseconds
# (2308.6) and a 0-1 detector score: it flattens every score to 0.9 and rounds a p90 of
# 0.96 up to 1.0, i.e. above the row's own max.
_SUMMARY_SIG_DIGITS = 4


def _decimals(hi: float) -> int:
    """Decimals giving ~_SUMMARY_SIG_DIGITS significant digits for a column topping out
    at ``hi``. Uniform within a row, so rounding stays monotonic: min <= p90 <= max holds
    after rounding exactly as it did before."""
    if hi <= 0:
        return 0
    return max(0, _SUMMARY_SIG_DIGITS - 1 - math.floor(math.log10(hi)))


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
                "SELECT result_json FROM page_runs WHERE md5=? AND src=? AND dst=? AND engines=? AND opt_hash=?",
                (md5, src, dst, engines, oh),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_run(self, md5: str, src: str, dst: str, engines: str, oh: str, result: list[dict]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO page_runs(md5, src, dst, engines, opt_hash, result_json) VALUES (?,?,?,?,?,?)",
                (md5, src, dst, engines, oh, json.dumps(result, ensure_ascii=False)),
            )
            self._conn.commit()

    def clear(self) -> int:
        """Drop all cached page results so everything is recomputed fresh on next
        use. Returns the rows removed."""
        with self._lock:
            n = self._conn.execute("DELETE FROM page_runs").rowcount
            self._conn.commit()
        return n

    # --- processing stats (raw page + per-crop rows) ---
    def record_stats(self, *, page: dict, regions: list[dict], skip_translate: bool = False) -> None:
        """Insert one page row + its per-crop rows in a single transaction. RAW rows
        (not pre-aggregated) so any percentile is computable later. Serialized by
        self._lock, so it's safe when several pages record concurrently."""
        pcols = ("engines", "src", "dst", "md5", "regions", "raw_regions",
                 "decode_ms", "lockwait_ms", "detect_ms", "recognize_ms",
                 "semwait_ms", "translate_ms", "total_ms")
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO page_stats(skip_translate,{','.join(pcols)}) "
                f"VALUES ({','.join('?' * (len(pcols) + 1))})",
                (int(skip_translate), *(page.get(c) for c in pcols)),
            )
            page_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO region_stats(page_id,crop_w,crop_h,score,source_len,dest_len,recognize_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                [(page_id, r.get("crop_w"), r.get("crop_h"), r.get("score"),
                  r.get("source_len"), r.get("dest_len"), r.get("recognize_ms")) for r in regions],
            )
            self._conn.commit()

    def stats_summary(self, engines: str | None = None) -> dict:
        """Per-page and per-crop summaries: count + {mean,min,max,median,p90,p99} per
        numeric column. Benchmark (skip_translate) pages excluded. Empty -> count 0.

        ``combos`` lists the engine combos the stats hold, page-heaviest first, and is
        deliberately NOT filtered — it's the caller's picker, so choosing one combo must
        not shrink the list to that one. Read the summary one combo at a time: pooling
        them averages a 20ms recognizer with a 200s one and represents neither."""
        where, params = "skip_translate=0", []
        if engines:
            where += " AND engines=?"
            params.append(engines)
        with self._lock:
            combos = self._conn.execute(
                "SELECT engines, count(*) FROM page_stats WHERE skip_translate=0 "
                "GROUP BY engines ORDER BY 2 DESC"
            ).fetchall()
            pages = self._conn.execute(
                f"SELECT {','.join(_PAGE_COLS)} FROM page_stats WHERE {where}", params
            ).fetchall()
            regions = self._conn.execute(
                f"SELECT {','.join(_REGION_COLS)} FROM region_stats "
                f"WHERE page_id IN (SELECT id FROM page_stats WHERE {where})", params
            ).fetchall()
        return {"pages": self._summarize(pages, _PAGE_COLS),
                "regions": self._summarize(regions, _REGION_COLS),
                "combos": [{"engines": e, "pages": n} for e, n in combos]}

    @staticmethod
    def _summarize(rows: list, cols: tuple) -> dict:
        if not rows:
            return {"count": 0, "metrics": {}}
        metrics = {}
        for i, name in enumerate(cols):
            vals = [r[i] for r in rows if r[i] is not None]
            if not vals:
                continue
            nd = _decimals(max(vals))
            m = {"mean": round(statistics.mean(vals), nd), "min": round(min(vals), nd),
                 "max": round(max(vals), nd), "median": round(statistics.median(vals), nd)}
            if len(vals) >= 2:
                # 99 cut points: p90=q[89], p99=q[98]. method="inclusive" because these
                # rows ARE the whole record, not a sample of some larger population --
                # and the default "exclusive" extrapolates past the data on a small
                # column, putting p99 above the max right next to it in the table.
                q = statistics.quantiles(vals, n=100, method="inclusive")
                m["p90"], m["p99"] = round(q[89], nd), round(q[98], nd)
            metrics[name] = m
        return {"count": len(rows), "metrics": metrics}

    def clear_stats(self) -> int:
        """Drop all processing stats (both tables). Separate from clear() so wiping the
        page-result cache ('recompute') doesn't also erase the stats history."""
        with self._lock:
            n = self._conn.execute("DELETE FROM region_stats").rowcount
            n += self._conn.execute("DELETE FROM page_stats").rowcount
            self._conn.commit()
        return n


cache = Cache()
