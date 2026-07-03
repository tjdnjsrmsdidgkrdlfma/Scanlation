"""Tests for scanlation-rtdetr. Self-contained: they instantiate the plugin /
postprocess directly (no server core). test_rtdetr_postprocess is model-free;
test_rtdetr self-skips unless transformers+torch and the weights are present.
Run: python -m tests (from packages/scanlation-rtdetr/)."""
