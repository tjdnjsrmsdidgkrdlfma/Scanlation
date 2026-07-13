"""Fast test suite: ``python -m tests`` (from server/).

Runs every model-free module. The slow model smokes live in each engine package
(``packages/scanlation-<engine>/tests/``); run ``python -m tests`` from there —
they self-skip when the weights/package aren't present.
"""
from __future__ import annotations

import sys

from tests import (
    test_cache,
    test_catalog,
    test_contracts,
    test_geometry,
    test_gpus,
    test_idle_unload,
    test_inference_gate,
    test_logconfig,
    test_orchestrator,
    test_pipeline,
    test_recognize_pool,
    test_registry,
    test_routes_admin,
    test_routes_auth,
    test_routes_plugins,
    test_routes_run,
    test_routes_settings,
    test_state,
)
from tests.helpers import run_modules

if __name__ == "__main__":
    # Route modules keep the original test_routes.py order (they share one cached
    # TestClient); test_logconfig last: it reconfigures global logging.
    sys.exit(run_modules([
        test_contracts,
        test_geometry,
        test_inference_gate,
        test_recognize_pool,
        test_pipeline,
        test_state,
        test_cache,
        test_catalog,
        test_registry,
        test_gpus,
        test_orchestrator,
        test_idle_unload,
        test_routes_run,
        test_routes_settings,
        test_routes_plugins,
        test_routes_admin,
        test_routes_auth,
        test_logconfig,
    ]))
