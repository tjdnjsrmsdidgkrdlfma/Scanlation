"""RTDetrDetector — RT-DETRv2 comic/manga text detector (transformers).

Replaces the segmentation-mask ``ctd`` detector. The model
(ogkalu/comic-text-and-bubble-detector) classifies each region as
bubble / text_bubble / text_free; for OCR we keep only the text regions
(text_bubble + text_free), drop the whole-bubble container, and dedup the
NMS-free overlaps RT-DETR leaves behind. Tuned defaults (conf 0.6, dedup
0.6/0.85) come from the tools/compare_models.py bake-off. The geometry lives in
``postprocess`` (model-free, unit-tested); this file is just load + inference.

torch/transformers are imported lazily so the module imports (and the registry
discovers the class) with no heavy deps. Weights are a HF transformers snapshot
(safetensors) fetched explicitly by install() into ``<models>/rtdetr``; load()
never downloads (``local_files_only``), matching the project's explicit-install
rule.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

from scanlation_sdk.context import context
from scanlation_sdk.contracts import Region
from scanlation_sdk.local_engine import LocalModelEngineBase
from . import postprocess

logger = logging.getLogger("scanlation.rtdetr")

# Tuned sweet spot (bake-off). Single source: OPTION_SCHEMA defaults AND detect()
# fallbacks both read this, so they can't drift.
DEFAULTS = {"conf": 0.6, "nms_iou": 0.6, "contain_thresh": 0.85}


class RTDetrDetector(LocalModelEngineBase):
    name = "rtdetr"
    display_name = "comic-text-and-bubble-detector"
    homepage = "https://huggingface.co/ogkalu/comic-text-and-bubble-detector"
    description = "RT-DETRv2 (ogkalu/comic-text-and-bubble-detector) comic/manga text & bubble detector. Runs on CPU. 172MB."
    OPTION_SCHEMA = {
        "conf": {"type": float, "default": DEFAULTS["conf"], "description": "Confidence threshold; raise to drop weak/noise boxes."},
        "nms_iou": {"type": float, "default": DEFAULTS["nms_iou"], "description": "Drop a box overlapping a kept one past this IoU (1.0 = off)."},
        "contain_thresh": {"type": float, "default": DEFAULTS["contain_thresh"], "description": "Drop a box this fraction nested inside a kept one (IoS; 1.0 = off)."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    REPO = "ogkalu/comic-text-and-bubble-detector"
    KEEP_LABELS = {"text_bubble", "text_free"}  # drop the whole-"bubble" container box
    WEIGHT_PATTERNS = ["config.json", "preprocessor_config.json", "model.safetensors"]
    INSTALL_HINT = (
        'Install first: POST /install_plugins/ {"rtdetr": true}, or '
        "`python tools/install.py rtdetr`, or set SCANLATION_RTDETR_MODEL=/path/to/model_dir."
    )

    def __init__(self) -> None:
        self._model = None
        self._proc = None
        self._device = "cpu"

    # --- weights / install ---
    def _model_dir(self) -> Path:
        env = os.environ.get("SCANLATION_RTDETR_MODEL")
        return Path(env) if env else context.models_dir / "rtdetr"

    def is_installed(self) -> bool:
        d = self._model_dir()
        return (d / "config.json").is_file() and (d / "model.safetensors").is_file()

    def _download(self) -> None:
        """Download the RT-DETR transformers weights (~172MB) into <models>/rtdetr.
        Fetches only the transformers files (safetensors + config + preprocessor),
        not the repo's separate ONNX."""
        from huggingface_hub import snapshot_download

        d = self._model_dir()
        d.mkdir(parents=True, exist_ok=True)
        logger.info("installing RT-DETR weights from %s -> %s", self.REPO, d)
        snapshot_download(self.REPO, local_dir=str(d), allow_patterns=self.WEIGHT_PATTERNS)
        logger.info("RT-DETR weights installed -> %s", d)

    def _load(self, device: str) -> None:
        from transformers import AutoImageProcessor  # lazy

        d = self._model_dir()
        self._device = device
        self._proc = AutoImageProcessor.from_pretrained(str(d), local_files_only=True)
        try:  # Auto covers most detection models
            from transformers import AutoModelForObjectDetection
            model = AutoModelForObjectDetection.from_pretrained(str(d), local_files_only=True)
        except (ValueError, KeyError):  # rt_detr_v2 on older transformers: name the class
            from transformers import RTDetrV2ForObjectDetection
            model = RTDetrV2ForObjectDetection.from_pretrained(str(d), local_files_only=True)
        self._model = model.to(self._device).eval()
        logger.info("RT-DETR loaded from %s on %s", d.name, self._device)

    def _unload(self) -> None:
        self._model = None
        self._proc = None

    # --- inference ---
    def detect(self, image: Image.Image, options: dict[str, Any]) -> list[Region]:
        if self._model is None:
            self.load()
        import torch

        conf = float(options.get("conf", DEFAULTS["conf"]))
        nms_iou = float(options.get("nms_iou", DEFAULTS["nms_iou"]))
        contain_thresh = float(options.get("contain_thresh", DEFAULTS["contain_thresh"]))

        img = image.convert("RGB")
        inputs = self._proc(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        sizes = torch.tensor([img.size[::-1]]).to(self._device)  # (height, width)
        r = self._proc.post_process_object_detection(out, target_sizes=sizes, threshold=conf)[0]
        id2label = getattr(self._model.config, "id2label", {})
        dets = [
            postprocess.Det(tuple(float(v) for v in box.tolist()),
                            id2label.get(int(lab), str(int(lab))), float(sc))
            for sc, lab, box in zip(r["scores"], r["labels"], r["boxes"])
        ]
        dets = postprocess.filter_labels(dets, self.KEEP_LABELS)  # drop bubble containers
        dets = postprocess.dedup(dets, nms_iou, contain_thresh)   # kill NMS-free overlaps
        return postprocess.to_regions(dets)
