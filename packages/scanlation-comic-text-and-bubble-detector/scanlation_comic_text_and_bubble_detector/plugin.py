"""ComicTextAndBubbleDetector — RT-DETRv2 comic/manga text detector (transformers).

The model (ogkalu/comic-text-and-bubble-detector) classifies each region as
bubble / text_bubble / text_free; for recognition we keep only the text regions
(text_bubble + text_free), drop the whole-bubble container, and dedup the
NMS-free overlaps RT-DETR leaves behind. Tuned defaults (conf 0.6, dedup
0.6/0.85) come from the tools/compare_models.py bake-off. The geometry lives in
``postprocess`` (model-free, unit-tested); this file is just load + inference.

torch/transformers are imported lazily so the module imports (and the registry
discovers the class) with no heavy deps. Weights are a HF transformers snapshot
(safetensors) fetched explicitly by install() into ``<models>/comic-text-and-bubble-detector``; load()
never downloads (``local_files_only``), matching the project's explicit-install
rule.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from scanlation_sdk.context import context
from scanlation_sdk.contracts import Region
from scanlation_sdk.local_engine import LocalModelEngineBase, install_hint, to_rgb
from . import postprocess


class ComicTextAndBubbleDetector(LocalModelEngineBase):
    name = "comic-text-and-bubble-detector"
    display_name = "comic-text-and-bubble-detector"
    homepage = "https://huggingface.co/ogkalu/comic-text-and-bubble-detector"
    description = "RT-DETRv2 (ogkalu/comic-text-and-bubble-detector) comic/manga text & bubble detector. Runs on CPU. 172MB."
    # Tuned sweet spot from the tools/compare_models.py bake-off. OPTION_SCHEMA is
    # the single source of these defaults; detect() reads the resolved options.
    OPTION_SCHEMA = {
        "conf": {"type": float, "default": 0.6, "description": "Confidence threshold; raise to drop weak/noise boxes."},
        "nms_iou": {"type": float, "default": 0.6, "description": "Drop a box overlapping a kept one past this IoU (1.0 = off)."},
        "contain_thresh": {"type": float, "default": 0.85, "description": "Drop a box this fraction nested inside a kept one (IoS; 1.0 = off)."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    REPO = "ogkalu/comic-text-and-bubble-detector"
    KEEP_LABELS = {"text_bubble", "text_free"}  # drop the whole-"bubble" container box
    WEIGHT_PATTERNS = ["config.json", "preprocessor_config.json", "model.safetensors"]
    INSTALL_HINT = install_hint(
        "comic-text-and-bubble-detector",
        extra=", or set SCANLATION_COMIC_TEXT_AND_BUBBLE_DETECTOR_MODEL=/path/to/model_dir.",
    )

    def __init__(self) -> None:
        self._model = None
        self._proc = None

    # --- weights / install ---
    def _model_dir(self) -> Path:
        env = os.environ.get("SCANLATION_COMIC_TEXT_AND_BUBBLE_DETECTOR_MODEL")
        return Path(env) if env else context.models_dir / "comic-text-and-bubble-detector"

    def is_installed(self) -> bool:
        d = self._model_dir()
        return (d / "config.json").is_file() and (d / "model.safetensors").is_file()

    def _download(self) -> None:
        """Download the RT-DETR transformers weights (~172MB) into <models>/comic-text-and-bubble-detector.
        Fetches only the transformers files (safetensors + config + preprocessor),
        not the repo's separate ONNX."""
        from huggingface_hub import snapshot_download

        d = self._model_dir()
        d.mkdir(parents=True, exist_ok=True)
        self._log.info("installing RT-DETR weights from %s -> %s", self.REPO, d)
        snapshot_download(self.REPO, local_dir=str(d), allow_patterns=self.WEIGHT_PATTERNS)
        self._log.info("RT-DETR weights installed -> %s", d)

    def _load(self, device: str) -> None:
        from transformers import AutoImageProcessor  # lazy

        d = self._model_dir()
        self._proc = AutoImageProcessor.from_pretrained(str(d), local_files_only=True)
        try:  # Auto covers most detection models
            from transformers import AutoModelForObjectDetection
            model = AutoModelForObjectDetection.from_pretrained(str(d), local_files_only=True)
        except (ValueError, KeyError):  # rt_detr_v2 on older transformers: name the class
            from transformers import RTDetrV2ForObjectDetection
            model = RTDetrV2ForObjectDetection.from_pretrained(str(d), local_files_only=True)
        # detect() reads self._device (set by LocalModelEngineBase.load after this returns);
        # inside _load the resolved device is the `device` arg.
        self._model = model.to(device).eval()

    def _unload(self) -> None:
        self._model = None
        self._proc = None

    # --- inference ---
    def detect(self, image: Image.Image, options: dict[str, Any]) -> list[Region]:
        import torch

        options = self.resolve_options(options)
        conf = options["conf"]
        nms_iou = options["nms_iou"]
        contain_thresh = options["contain_thresh"]

        img = to_rgb(image)
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
