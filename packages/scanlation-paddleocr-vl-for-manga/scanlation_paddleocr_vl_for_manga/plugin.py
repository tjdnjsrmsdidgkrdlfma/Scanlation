"""PaddleOcrVLForMangaRecognizer — PaddleOCR-VL (manga fine-tune) text recognizer.

A vision-language OCR: it reads an upright crop end-to-end (the pipeline has
already deskewed the crop, so no rotation here). Wraps
jzhang533/PaddleOCR-VL-For-Manga — the bake-off's most accurate recognizer (88%):
strong on numbers/English/symbols AND pure Japanese, with no weak category.

GPU-intended: ~1s/crop on GPU (bf16, ~2-4GB VRAM); CPU works but ~60s/crop
(0.9B autoregressive VLM = memory-bandwidth bound). Its DEFAULT_DEVICE is cuda,
overridable per-engine in /admin. Keep manga-ocr as the CPU-viable recognizer;
this is the accuracy option to select once a GPU is available.

The fine-tune ships no readable processor, so the processor is loaded from the
base repo (PaddlePaddle/PaddleOCR-VL). Native transformers 5.x path
(AutoModelForImageTextToText, NO trust_remote_code). Heavy deps + the weight
download are deferred to install()/load().
"""
from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image

from scanlation_sdk.contracts import Region
from scanlation_sdk.local_engine import LocalModelEngineBase, downscale_to_cap, install_hint, to_rgb

logger = logging.getLogger("scanlation.PaddleOCR-VL-For-Manga")

_MODES = ("area", "box", "grid28", "boxgrid", "pow2")  # downscale_mode choices; validated in recognize


class PaddleOcrVLForMangaRecognizer(LocalModelEngineBase):
    name = "PaddleOCR-VL-For-Manga"
    display_name = "PaddleOCR-VL-For-Manga"
    homepage = "https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga"
    description = "PaddleOCR-VL manga fine-tune (0.9B VLM). Best accuracy. Needs a GPU. 1.8GB."
    DEFAULT_DEVICE = "cuda"  # GPU-intended (CPU ~60s/crop); overridable per-engine in /admin
    OPTION_SCHEMA = {
        "max_new_tokens": {"type": int, "default": 1024,
                           "description": "Max output tokens per crop; lower to cap runaway generation."},
        "max_pixels": {"type": int,
                       "default": int(os.environ.get("SCANLATION_RECOGNIZE_MAX_PIXELS", "150000")),
                       "description": "Downscale crops above this many pixels before OCR to cut vision tokens (~1.66x). 0 = off."},
        "downscale_mode": {"type": str,
                           "default": os.environ.get("SCANLATION_RECOGNIZE_DOWNSCALE_MODE", "pow2"),
                           "description": "How to downscale when max_pixels applies: pow2 (recommended) / box / area / grid28 / boxgrid."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    PROC_REPO = "PaddlePaddle/PaddleOCR-VL"  # the fine-tune's own processor is a 4.x format -> load base's
    PROMPT = "OCR:"
    INSTALL_HINT = install_hint("PaddleOCR-VL-For-Manga")

    def __init__(self) -> None:
        self._model = None
        self._proc = None

    # --- weights / install ---
    def _repo(self) -> str:
        """The weights repo id (or a local dir), env-overridable."""
        return os.environ.get("SCANLATION_PADDLEOCR_VL_FOR_MANGA_MODEL") or "jzhang533/PaddleOCR-VL-For-Manga"

    def is_installed(self) -> bool:
        repo = self._repo()
        if os.path.isdir(repo):  # local-dir override
            return True
        try:
            from huggingface_hub import try_to_load_from_cache

            weights = try_to_load_from_cache(repo, "config.json")
            proc = try_to_load_from_cache(self.PROC_REPO, "preprocessor_config.json")
            return isinstance(weights, str) and isinstance(proc, str)
        except Exception:  # noqa: BLE001
            return False

    def _download(self) -> None:
        """Download the ~1.8GB fine-tune weights + the base processor into the HF
        cache."""
        from huggingface_hub import snapshot_download

        repo = self._repo()
        logger.info("installing PaddleOCR-VL weights %s + processor %s", repo, self.PROC_REPO)
        if not os.path.isdir(repo):
            snapshot_download(repo)
        snapshot_download(self.PROC_REPO)  # processor files only (~13MB); base weights not needed
        logger.info("PaddleOCR-VL installed")

    def _load(self, device: str) -> None:
        import torch  # lazy
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._proc = AutoProcessor.from_pretrained(self.PROC_REPO, local_files_only=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._repo(),
            torch_dtype=(torch.float32 if device == "cpu" else "auto"),  # fp16 on CPU is a trap; GPU picks bf16
            device_map=device,
            local_files_only=True,  # contract: load() never downloads
        ).eval()

    def _unload(self) -> None:
        self._model = None
        self._proc = None

    # --- inference ---
    def _prompt(self) -> str:
        """The chat-template prompt string (image placeholder + PROMPT), shared by the
        single- and batch-recognize paths."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}]
        return self._proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _cap(self, crop: Image.Image, options: dict[str, Any]) -> Image.Image:
        """Downscale a crop to the vision-token cap (max_pixels / downscale_mode)."""
        mode = options["downscale_mode"] if options["downscale_mode"] in _MODES else "pow2"
        return downscale_to_cap(to_rgb(crop), options["max_pixels"], mode)

    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        options = self.resolve_options(options)
        crop = self._cap(crop, options)
        inputs = self._proc(text=[self._prompt()], images=[crop], return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=options["max_new_tokens"], do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()

    def recognize_batch(self, crops: list[Image.Image], regions: list[Region],
                        options: dict[str, Any]) -> list[str]:
        """Read a whole page's crops in ONE batched generate (the BatchRecognizer
        seam the pipeline uses). Caps each crop, then buckets largest-first so the
        ragged vision-token lengths need the least left-pad — a padding/position_ids
        mismatch is the 'silently wrong' failure mode for a dynamic-res VLM, and
        least-pad minimises it (packages/scanlation-server/tools/recognize-crop-batching.md).
        Decodes back to input order; one string per crop."""
        if not crops:
            return []
        options = self.resolve_options(options)
        capped = [self._cap(c, options) for c in crops]
        order = sorted(range(len(capped)), key=lambda i: capped[i].width * capped[i].height, reverse=True)
        self._proc.tokenizer.padding_side = "left"  # gen frontier aligns at the right edge
        inputs = self._proc(text=[self._prompt()] * len(capped), images=[capped[i] for i in order],
                            padding=True, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=options["max_new_tokens"], do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = [self._proc.decode(g, skip_special_tokens=True).strip() for g in gen]
        result = [""] * len(crops)
        for k, i in enumerate(order):
            result[i] = decoded[k]
        return result
