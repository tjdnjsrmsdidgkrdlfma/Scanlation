"""LocalModelEngineBase — shared lifecycle for engines whose weights live
locally and load into torch (rtdetr/mangaocr/paddleocrvl).

Centralizes what those engines genuinely duplicate: the load/install guards,
the "install() downloads, load() never does" project rule, device selection,
and CUDA cache release on unload. ``is_installed()`` stays per-engine — the
checks (local files vs HF cache probe) genuinely differ.

Plugin-facing only: the server core must not import this module (see device.py).
"""
from __future__ import annotations

from scanlation_sdk.contracts import EngineBase
from scanlation_sdk.device import pick_device, release_cuda_cache


class LocalModelEngineBase(EngineBase):
    # Per-engine "how to install" tail of the not-installed error.
    INSTALL_HINT: str = ""
    # Class default so a subclass that skips super().__init__() is still safe
    # (engine_meta.safe_is_installed instantiates throwaway cls()).
    _loaded: bool = False
    # Compute device this engine loads onto when the user sets no override —
    # the code default, like an OPTION_SCHEMA option's `default`. Subclasses
    # override (paddleocrvl -> "cuda"); cpu-viable engines keep "cpu".
    DEFAULT_DEVICE: str = "cpu"
    # Per-engine device override injected by the registry from admin state;
    # None -> DEFAULT_DEVICE. Class default keeps super().__init__()-skipping
    # subclasses safe, same as _loaded.
    _device_override: str | None = None

    # --- subclass hooks ---
    def _download(self) -> None:
        """install()'s body: fetch weights (snapshot_download etc.)."""
        raise NotImplementedError

    def _load(self, device: str) -> None:
        """Acquire model/processor attributes on ``device``."""
        raise NotImplementedError

    def _unload(self) -> None:
        """Drop whatever _load() set."""
        raise NotImplementedError

    # --- shared lifecycle ---
    def install(self) -> None:
        if self.is_installed():
            return
        self._download()

    def load(self) -> None:
        if self._loaded:
            return
        if not self.is_installed():
            raise RuntimeError(f"{self.name} weights not installed. {self.INSTALL_HINT}")
        self._load(pick_device(self._device_override or self.DEFAULT_DEVICE))
        self._loaded = True

    def unload(self) -> None:
        self._unload()
        self._loaded = False
        release_cuda_cache()
