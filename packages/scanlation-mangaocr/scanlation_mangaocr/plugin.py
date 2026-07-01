"""MangaOcrRecognizer — manga-ocr (kha-white) Japanese text recognizer.

A pure recognizer: it expects an upright crop and reads it (vertical and
horizontal Japanese natively). The pipeline has already deskewed the crop, so
no rotation happens here. Heavy deps (torch/transformers) and the model
download are deferred to load().
"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from scanlation_sdk.context import context
from scanlation_sdk.contracts import EngineBase, Region

logger = logging.getLogger("scanlation.mangaocr")


class MangaOcrRecognizer(EngineBase):
    name = "mangaocr"
    display_name = "manga-ocr"
    homepage = "https://github.com/kha-white/manga-ocr"
    description = "Japanese manga text recognizer (ViT-encoder/BERT-decoder). Reads vertical + horizontal natively."
    warning = "Downloads ~400MB model (kha-white/manga-ocr-base) on first use."
    OPTION_SCHEMA: dict = {}
    SUPPORTED_SRC = ["ja"]

    MODEL_REPO = "kha-white/manga-ocr-base"

    def __init__(self) -> None:
        self._m = None

    def is_installed(self) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache

            return isinstance(try_to_load_from_cache(self.MODEL_REPO, "config.json"), str)
        except Exception:  # noqa: BLE001
            return False

    def install(self) -> None:
        """Download the model (~400MB) into the HF cache. Explicit — not from load()."""
        from huggingface_hub import snapshot_download

        logger.info("installing manga-ocr model %s", self.MODEL_REPO)
        snapshot_download(self.MODEL_REPO)
        logger.info("manga-ocr model installed")

    def load(self) -> None:
        if self._m is not None:
            return
        if not self.is_installed():
            raise RuntimeError(
                'manga-ocr model not installed. Install first: POST /manage_plugins/ {"mangaocr": true}, '
                "or `python tools/install.py mangaocr`."
            )
        from manga_ocr import MangaOcr  # lazy: torch + transformers

        # force CPU only when the device hint is cpu; rocm/cuda lets torch pick.
        self._m = MangaOcr(force_cpu=(context.device.lower() == "cpu"))
        logger.info("manga-ocr loaded (force_cpu=%s)", context.device.lower() == "cpu")

    def unload(self) -> None:
        self._m = None

    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        if self._m is None:
            self.load()
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        return self._m(crop)
