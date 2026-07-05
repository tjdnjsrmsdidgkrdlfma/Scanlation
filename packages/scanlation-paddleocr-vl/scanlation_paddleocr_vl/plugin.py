"""PaddleOcrVLRecognizer — PaddleOCR-VL (manga fine-tune) text recognizer.

A vision-language OCR: it reads an upright crop end-to-end (the pipeline has
already deskewed the crop, so no rotation here). Wraps
jzhang533/PaddleOCR-VL-For-Manga — the bake-off's most accurate recognizer (88%):
strong on numbers/English/symbols AND pure Japanese, with no weak category.

GPU-intended: ~1s/crop on GPU (bf16, ~2-4GB VRAM); CPU works but ~60s/crop
(0.9B autoregressive VLM = memory-bandwidth bound). It honors the shared
``context.device`` switch like the other engines, so one /set_device moves the
whole pipeline. Keep mangaocr as the CPU-viable recognizer; this is the accuracy
option to select once a GPU is available.

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
from scanlation_sdk.local_engine import LocalModelEngineBase

logger = logging.getLogger("scanlation.paddleocrvl")


class PaddleOcrVLRecognizer(LocalModelEngineBase):
    name = "paddleocrvl"
    display_name = "PaddleOCR-VL-For-Manga"
    homepage = "https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga"
    description = "PaddleOCR-VL manga fine-tune (0.9B VLM). Best bake-off accuracy — reads numbers/English/symbols + Japanese."
    warning = "Downloads ~1.8GB weights on install; needs torch + transformers>=5. GPU strongly recommended (CPU ~60s/crop)."
    OPTION_SCHEMA = {
        "max_new_tokens": {"type": int, "default": 1024,
                           "description": "Max output tokens per crop; lower to cap runaway generation."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    PROC_REPO = "PaddlePaddle/PaddleOCR-VL"  # the fine-tune's own processor is a 4.x format -> load base's
    PROMPT = "OCR:"
    INSTALL_HINT = (
        'Install first: POST /install_plugins/ {"paddleocrvl": true}, or '
        "`python tools/install.py paddleocrvl`."
    )

    def __init__(self) -> None:
        self._model = None
        self._proc = None

    # --- weights / install ---
    def _repo(self) -> str:
        """The weights repo id (or a local dir), env-overridable like rtdetr."""
        return os.environ.get("SCANLATION_PADDLEOCRVL_MODEL") or "jzhang533/PaddleOCR-VL-For-Manga"

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
        logger.info("PaddleOCR-VL loaded on %s", device)

    def _unload(self) -> None:
        self._model = None
        self._proc = None

    # --- inference ---
    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        if self._model is None:
            self.load()
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}]
        text = self._proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._proc(text=[text], images=[crop], return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs, max_new_tokens=int(options.get("max_new_tokens", 1024)), do_sample=False,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()
