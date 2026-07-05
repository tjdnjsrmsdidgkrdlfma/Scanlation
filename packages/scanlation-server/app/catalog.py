"""The catalog of *installable* engines — a small static manifest.

It can't come from entry_points (those only list *installed* engines) nor from
the source (the core image ships none), so the set of engines /admin can offer
to install is hardcoded here. Installed engines are still discovered purely via
entry_points in the registry; this manifest only drives the install UI.

Separated from ``plugins_install`` (the pip/weights machinery) so the "what can
be installed" data and the "how to install it" logic don't sit in one file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# name = registry/engine name; package = pip/dist name (and the packages/<package>
# subdir). Installed engines are found via entry_points; this only lists what
# /admin can offer to install.
_CATALOG: dict[str, dict] = {
    "comic-text-and-bubble-detector": {
        "package": "scanlation-comic-text-and-bubble-detector",
        "display_name": "comic-text-and-bubble-detector",
        "roles": ["detector"],
        "description": "RT-DETRv2 (ogkalu/comic-text-and-bubble-detector) comic/manga text & bubble detector. Runs on CPU. 172MB.",
        # steer torch to the CPU wheel by default (like manga-ocr) so a Docker
        # one-click install doesn't pull a giant CUDA wheel; a GPU torch build is
        # a host-level install.
        "pip_args": ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    },
    "manga-ocr": {
        "package": "scanlation-manga-ocr",
        "display_name": "Manga OCR",
        "roles": ["recognizer"],
        "description": "ViT-encoder/BERT-decoder Japanese OCR. Fast, solid accuracy. Runs on CPU. 400MB.",
        # steer torch to the CPU index (its +cpu local version outranks the plain
        # PyPI CUDA wheel, so pip prefers it).
        "pip_args": ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    },
    "PaddleOCR-VL-For-Manga": {
        "package": "scanlation-paddleocr-vl-for-manga",
        "display_name": "PaddleOCR-VL-For-Manga",
        "roles": ["recognizer"],
        "description": "PaddleOCR-VL manga fine-tune (0.9B VLM). Best accuracy. Needs a GPU. 1.8GB.",
        # CPU wheel by default like the other torch plugins; a GPU/ROCm torch build
        # is a host-level install (the model needs a GPU to be practical).
        "pip_args": ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    },
    "Ollama": {
        "package": "scanlation-ollama",
        "display_name": "Ollama",
        "roles": ["translator"],
        "description": "LLM translation via a local ollama server (must be running, model selected in /admin).",
        "pip_args": [],
    },
    "llama.cpp": {
        "package": "scanlation-llama-cpp",
        "display_name": "llama.cpp",
        "roles": ["translator"],
        "description": "LLM translation via an OpenAI-compatible /v1 server (llama.cpp, vllm, LM Studio…; must be running, model selected in /admin).",
        "pip_args": [],
    },
}


@dataclass
class CatalogEntry:
    name: str                       # engine name = registry key (e.g. "comic-text-and-bubble-detector")
    package: str                    # pip/dist name (e.g. "scanlation-comic-text-and-bubble-detector")
    display_name: str = ""          # human-readable name shown before install
    description: str = ""
    roles: list[str] = field(default_factory=list)
    pip_args: list[str] = field(default_factory=list)


def catalog() -> dict[str, CatalogEntry]:
    """The static manifest of installable plugins, keyed by engine name."""
    return {
        name: CatalogEntry(
            name=name,
            package=spec["package"],
            display_name=spec.get("display_name") or name,
            description=spec["description"],
            roles=list(spec["roles"]),
            pip_args=list(spec["pip_args"]),
        )
        for name, spec in _CATALOG.items()
    }
