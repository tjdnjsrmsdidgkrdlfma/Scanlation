"""Tests for scanlation-comic-text-and-bubble-detector. Self-contained: they
instantiate the plugin / postprocess directly (no server core).
test_comic_text_and_bubble_detector_postprocess is model-free;
test_comic_text_and_bubble_detector self-skips unless transformers+torch and the
weights are present. Run: python -m tests (from
packages/scanlation-comic-text-and-bubble-detector/)."""
