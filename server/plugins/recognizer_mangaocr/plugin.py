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

from app.config import settings
from app.contracts import EngineBase, Region

logger = logging.getLogger("scanlation.mangaocr")


class MangaOcrRecognizer(EngineBase):
    name = "mangaocr"
    display_name = "manga-ocr"
    homepage = "https://github.com/kha-white/manga-ocr"
    description = "Japanese manga text recognizer (ViT-encoder/BERT-decoder). Reads vertical + horizontal natively."
    warning = "Downloads ~400MB model (kha-white/manga-ocr-base) on first use."
    OPTION_SCHEMA: dict = {}
    SUPPORTED_SRC = ["ja"]

    def __init__(self) -> None:
        self._m = None

    def load(self) -> None:
        if self._m is not None:
            return
        from manga_ocr import MangaOcr  # lazy: torch + transformers

        # force CPU only when the device hint is cpu; rocm/cuda lets torch pick.
        self._m = MangaOcr(force_cpu=(settings.device.lower() == "cpu"))
        logger.info("manga-ocr loaded (force_cpu=%s)", settings.device.lower() == "cpu")

    def unload(self) -> None:
        self._m = None

    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        if self._m is None:
            self.load()
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        return self._m(crop)
