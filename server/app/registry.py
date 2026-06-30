"""Plugin discovery + lazy instantiation.

Two sources, merged: a hardcoded builtin map (so a plain source checkout with
no ``pip install`` still works) and ``importlib.metadata`` entry_points (so any
third-party package declaring the ``scanlation.<role>`` groups is auto-found).

Engines are instantiated lazily on first use — that's when VRAM/model weights
are actually loaded. Class-level metadata (OPTION_SCHEMA, description, ...) is
read without instantiating, so the handshake/options routes stay light.
"""
from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

ROLES: dict[str, str] = {
    "detector": "scanlation.detectors",
    "recognizer": "scanlation.recognizers",
    "translator": "scanlation.translators",
}

# Fallback for source checkouts (no install => no entry_points).
_BUILTIN: dict[str, dict[str, str]] = {
    "detector": {
        "dummy": "plugins.dummy.plugin:DummyDetector",
        "ctd": "plugins.detector_ctd.plugin:CTDDetector",
    },
    "recognizer": {
        "dummy": "plugins.dummy.plugin:DummyRecognizer",
        "mangaocr": "plugins.recognizer_mangaocr.plugin:MangaOcrRecognizer",
    },
    "translator": {
        "dummy": "plugins.dummy.plugin:DummyTranslator",
        "ollama": "plugins.translator_ollama.plugin:OllamaTranslator",
        "llamacpp": "plugins.translator_llamacpp.plugin:LlamaCppTranslator",
    },
}


def _load_class(path: str) -> type:
    module_name, _, cls_name = path.partition(":")
    return getattr(importlib.import_module(module_name), cls_name)


class Registry:
    def __init__(self) -> None:
        self._classes: dict[str, dict[str, type]] = {r: {} for r in ROLES}
        self._instances: dict[tuple[str, str], Any] = {}
        self._discover()

    def _discover(self) -> None:
        # 1) builtin fallback
        for role, mapping in _BUILTIN.items():
            for name, path in mapping.items():
                try:
                    self._classes[role][name] = _load_class(path)
                except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
                    pass
        # 2) entry_points (override/extend builtin)
        for role, group in ROLES.items():
            try:
                eps = entry_points(group=group)
            except TypeError:  # Python < 3.10 API
                eps = entry_points().get(group, [])
            for ep in eps:
                try:
                    self._classes[role][ep.name] = ep.load()
                except Exception:  # noqa: BLE001
                    pass

    # --- queries (no instantiation) ---
    def names(self, role: str) -> list[str]:
        return sorted(self._classes[role])

    def has(self, role: str, name: str) -> bool:
        return name in self._classes[role]

    def get_class(self, role: str, name: str) -> type:
        return self._classes[role][name]

    def all_classes(self) -> dict[str, dict[str, type]]:
        return self._classes

    # --- lazy instance (loads weights on first use) ---
    def get(self, role: str, name: str) -> Any:
        key = (role, name)
        if key not in self._instances:
            inst = self._classes[role][name]()
            inst.load()
            self._instances[key] = inst
        return self._instances[key]

    def unload(self, role: str, name: str) -> None:
        inst = self._instances.pop((role, name), None)
        if inst is not None:
            inst.unload()


registry = Registry()
