"""OCR+translate endpoints: /run_pipeline/ and /run_lookup/.

  * /run_lookup/  : client POSTs {md5, options} -> cache hit returns result,
                    miss = 200 {result: null} (a probe, not a 404 control signal)
  * /run_pipeline/: client POSTs {md5, contents} -> md5(base64) verified, runs
md5 is computed over the base64 *string* (not raw bytes) — mismatch => 400.

The orchestration (plan, cache identity, GPU-lock sequencing, in-flight dedup)
lives in ``app.orchestrator``; this module is just request validation and the
error -> status-code mapping.
"""
from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, HTTPException

from ..orchestrator import BadImageError, cached_result, make_plan, run_page
from ..registry import registry
from ..schemas import RunRequest

router = APIRouter()
logger = logging.getLogger("scanlation.run")


def _require(role: str, name: str) -> None:
    """400 if a role has no engine installed/selected (the core ships none)."""
    if not name or not registry.has(role, name):
        raise HTTPException(
            status_code=400,
            detail=f"no {role} engine installed — install and select one in /admin",
        )


@router.post("/run_lookup/")
def run_lookup(req: RunRequest) -> dict:
    """Read-only cache probe: ``{result: <cached list or null>}``, always 200.

    Lets the client skip re-uploading the image when the page is already cached
    (a bandwidth win), WITHOUT using a 404 as a control signal — a miss is a plain
    200 with ``result: null``. Same ``{result: ...}`` envelope as /run_pipeline/,
    so ``null`` (not cached) is distinct from ``[]`` (cached empty page). Body
    reuses RunRequest; only md5 + options are read (contents ignored)."""
    plan = make_plan(req.options)
    return {"result": cached_result(plan, req.md5)}  # None on miss


@router.post("/run_pipeline/")
async def run_pipeline(req: RunRequest) -> dict:
    plan = make_plan(req.options)

    # Contents are required — the cache probe moved to /run_lookup/ (a 200 lookup),
    # so run_pipeline is single-purpose: it always does the real work.
    if req.contents is None:
        raise HTTPException(
            status_code=400,
            detail="contents required — use /run_lookup/ to probe the cache",
        )

    # --- verify md5 over the base64 string ---
    if req.md5 != hashlib.md5(req.contents.encode("utf-8")).hexdigest():
        raise HTTPException(status_code=400, detail="md5 mismatch")

    if not req.force:
        cached = cached_result(plan, req.md5)
        if cached is not None:
            return {"result": cached}

    # Need a real engine per role to run (cache miss) -> 400 if any is missing.
    _require("detector", plan.detector)
    _require("recognizer", plan.recognizer)
    _require("translator", plan.translator)

    try:
        result = await run_page(plan, req.md5, req.contents)
    except BadImageError as exc:
        raise HTTPException(status_code=400, detail=f"bad image: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline failed md5=%s", req.md5[:8])
        raise HTTPException(status_code=500, detail=str(exc))
    return {"result": result}
