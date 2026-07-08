"""One-call logging setup for the server.

The app shipped with NO logging config, so ``scanlation.*`` INFO lines (e.g. the
engines' "loaded on cpu") were swallowed while manga-ocr's loguru output showed —
misleading. This centralizes it: a timestamped formatter, our namespace opened to
``SCANLATION_LOG_LEVEL`` (default INFO), and third-party libs (transformers/httpx)
kept at WARNING via the root so they don't drown the log.

uvicorn's default access log is disabled here — the request-timing middleware in
``app.main`` replaces it with a timestamped ``METHOD PATH -> STATUS Nms`` line
(and, since the cache probe moved to /run_lookup/, no more control-flow 404s).

Called once from the app lifespan (after uvicorn has set up its own logging, so
``dictConfig`` overrides it).
"""
from __future__ import annotations

import logging.config

# The level ``configure_logging`` opened ``scanlation.*`` to. ``apply_verbose(False)``
# returns here rather than to a hardcoded INFO, so turning the 동작-tab toggle off
# doesn't quietly override SCANLATION_LOG_LEVEL.
_base_level = "INFO"


def configure_logging(level: str = "INFO") -> None:
    global _base_level
    lvl = level.upper()
    _base_level = lvl
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "ts": {
                "format": "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "ts",
                "stream": "ext://sys.stderr",
            },
        },
        # root=WARNING gates third-party (transformers/httpx/hf) noise; our own
        # namespace is opened to `lvl` and propagates up to the console handler.
        "root": {"handlers": ["console"], "level": "WARNING"},
        "loggers": {
            "scanlation": {"level": lvl, "propagate": True},          # app + engine plugins
            "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
            # silenced: our timing middleware is the access log instead.
            "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": False},
        },
    })


def apply_verbose(on: bool) -> None:
    """Flip the app's own logger (scanlation.*) to DEBUG for the verbose
    per-detection/translation detail, or back to the level ``configure_logging``
    opened it to — at runtime, no reconfigure. Called by the /admin 동작 toggle and
    once at startup from the persisted state, right after configure_logging."""
    logging.getLogger("scanlation").setLevel(logging.DEBUG if on else _base_level)
