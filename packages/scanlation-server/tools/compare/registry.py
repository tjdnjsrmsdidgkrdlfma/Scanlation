"""The adapter registry: every candidate model, and selection by kind/id."""
from __future__ import annotations

from compare.adapters import (
    HfCausalVlmAdapter, HfVlmAdapter, MangaOcrAdapter, MitOcrAdapter,
    OllamaVlmAdapter, TransformersDetAdapter, UltralyticsDetAdapter,
)
from compare.core import Adapter


# --------------------------------------------------------------------------- #
# adapter registry  (repo ids / tags / classes verified 2026-07 against HF cards
# + ollama library; see the research notes. `heavy` VLMs are multi-GB downloads
# so they only run when named in --only, never by accident on a bare `ocr` run.)
# --------------------------------------------------------------------------- #
def all_adapters() -> list[Adapter]:
    return [
        # --- detectors ---
        TransformersDetAdapter("ogkalu_rtdetr", "ogkalu/comic-text-and-bubble-detector",
                               "ogkalu RT-DETRv2 (text_bubble+text_free)",
                               keep_labels={"text_bubble", "text_free"},
                               conf=0.6, nms_iou=0.6, contain_thresh=0.85),  # conf=0.6 = tuned sweet spot
        UltralyticsDetAdapter("kiuyha_yolo", "Kiuyha/Manga-Bubble-YOLO",
                              "Kiuyha Manga-Bubble-YOLO26 (needs recent ultralytics)", imgsz=1280),
        UltralyticsDetAdapter("ogkalu_yolov8m", "ogkalu/comic-speech-bubble-detector-yolov8m",
                              "ogkalu speech-bubble YOLOv8m", imgsz=1024),
        UltralyticsDetAdapter("kitsumed_seg", "kitsumed/yolov8m_seg-speech-bubble",
                              "kitsumed YOLOv8m-seg (mask)", seg=True, imgsz=1024),
        # --- ocr ---
        MangaOcrAdapter(),
        OllamaVlmAdapter("qwen3vl", "qwen3-vl:4b-instruct", "Qwen3-VL 4B Instruct (ollama, non-thinking)"),
        HfCausalVlmAdapter("dots_ocr", "rednote-hilab/dots.ocr", "dots.ocr",
                           "Extract the text content from this image.",
                           needs=("transformers", "torch", "qwen_vl_utils")),
        HfVlmAdapter("paddleocr_vl", "PaddlePaddle/PaddleOCR-VL", "PaddleOCR-VL (0.9B, native)",
                     prompt="OCR:", max_new_tokens=1024),  # native transformers 5.x impl (remote code is upstream-broken)
        HfVlmAdapter("paddleocr_manga", "jzhang533/PaddleOCR-VL-For-Manga",
                     "PaddleOCR-VL manga fine-tune (jzhang533)", prompt="OCR:", max_new_tokens=1024,
                     processor_repo="PaddlePaddle/PaddleOCR-VL"),  # native weights + base processor (its own is 4.x-format)
        MitOcrAdapter("mit_48px_ctc", "ctc", "48px_ctc (manga-image-translator, CTC)"),
        MitOcrAdapter("mit_48px", "48px", "48px (manga-image-translator, attention)"),
    ]


def _select(kind: str, only: str | None, exclude: str | None = None) -> list[Adapter]:
    ads = [a for a in all_adapters() if a.kind == kind]
    if only:
        want = {s.strip() for s in only.split(",")}
        ads = [a for a in ads if a.id in want]
    if exclude:
        drop = {s.strip() for s in exclude.split(",")}
        ads = [a for a in ads if a.id not in drop]
    return ads
