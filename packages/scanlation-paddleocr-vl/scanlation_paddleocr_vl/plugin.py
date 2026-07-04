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

from scanlation_sdk.context import context
from scanlation_sdk.contracts import EngineBase, Region

logger = logging.getLogger("scanlation.paddleocrvl")


class PaddleOcrVLRecognizer(EngineBase):
    name = "paddleocrvl"
    display_name = "PaddleOCR-VL (manga)"
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

    def install(self) -> None:
        """Download the ~1.8GB fine-tune weights + the base processor into the HF
        cache. Explicit — never called by load()."""
        from huggingface_hub import snapshot_download

        repo = self._repo()
        logger.info("installing PaddleOCR-VL weights %s + processor %s", repo, self.PROC_REPO)
        if not os.path.isdir(repo):
            snapshot_download(repo)
        snapshot_download(self.PROC_REPO)  # processor files only (~13MB); base weights not needed
        logger.info("PaddleOCR-VL installed")

    @staticmethod
    def _pick_device() -> str:
        """Honor context.device (the /admin-set, SCANLATION_DEVICE-seeded hint):
        'cpu' pins CPU; anything else means GPU — cuda if actually available, else
        CPU fallback. Same switch as rtdetr/mangaocr (a ROCm torch build reports cuda)."""
        if context.device.lower() == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_installed():
            raise RuntimeError(
                'PaddleOCR-VL weights not installed. Install first: POST /install_plugins/ {"paddleocrvl": true}, '
                "or `python tools/install.py paddleocrvl`."
            )
        import torch  # lazy
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dev = self._pick_device()
        self._proc = AutoProcessor.from_pretrained(self.PROC_REPO, local_files_only=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._repo(),
            torch_dtype=(torch.float32 if dev == "cpu" else "auto"),  # fp16 on CPU is a trap; GPU picks bf16
            device_map=dev,
            local_files_only=True,  # contract: load() never downloads
        ).eval()
        logger.info("PaddleOCR-VL loaded on %s", dev)

    def unload(self) -> None:
        self._model = None
        self._proc = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

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
