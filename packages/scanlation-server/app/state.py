"""Mutable selection state + runtime concurrency primitives.

Selection (which engine per role, src/dst langs, per-engine option overrides) is
persisted to a small json so restarts keep the user's choice. The GPU lock and
in-flight dedupe map are runtime-only.
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import settings


class InferenceGate:
    """Bounded-concurrency reader/writer gate over the detect+recognize half.

    Replaces the single ``gpu_lock`` mutex. Up to K *readers* (inference halves)
    run at once, so several images' crops fan out into the shared RecognizePool
    together (cross-image overlap) instead of one image at a time. A *writer*
    (lifecycle mutation: device/worker-count change, idle-unload) drains all K
    permits for exclusivity against every in-flight reader — the invariant the old
    single lock provided. K=1 is byte-identical to a single mutex (one reader at a
    time; a writer takes the lone permit). K is per-recognizer (see
    ``resolve_gpu_concurrency``); the gate is rebuilt when it changes.

    asyncio primitives bind to the loop that first awaits them, so a fresh gate is
    created per event loop where needed (tests; the live server is one loop)."""

    def __init__(self, k: int) -> None:
        self._k = max(1, int(k))
        self._slots = asyncio.Semaphore(self._k)   # reader permits
        self._writer = asyncio.Lock()              # writer serialization + new-reader gate

    @asynccontextmanager
    async def reader(self):
        # Hold _writer only long enough to grab a slot, so "no writer AND a permit"
        # is atomic; release it immediately so K readers run concurrently.
        async with self._writer:
            await self._slots.acquire()
        try:
            yield
        finally:
            self._slots.release()

    @asynccontextmanager
    async def writer(self):
        async with self._writer:                   # serialize writers (no deadlock)
            for _ in range(self._k):
                await self._slots.acquire()         # drain every in-flight reader
            try:
                yield
            finally:
                for _ in range(self._k):
                    self._slots.release()


@dataclass
class Selection:
    detector: str = settings.default_detector
    recognizer: str = settings.default_recognizer
    translator: str = settings.default_translator
    lang_src: str = settings.default_lang_src
    lang_dst: str = settings.default_lang_dst
    # {engine_name: {opt: val}} overrides applied on top of schema defaults.
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    # {engine_name: "cpu"|"cuda"|"cuda:N"} per-engine compute-device override (N =
    # GPU index). Absent -> the engine's DEFAULT_DEVICE (its code default). Only
    # in-process engines (detector + recognizer) honor it; LLM engines are separate
    # processes and ignore it.
    devices: dict[str, str] = field(default_factory=dict)
    # {engine_name: N} per-engine recognize worker-pool size (process count). Like
    # ``devices`` this is a LOAD-TIME property (the pool is built with N workers, not
    # changed per crop), so it lives here rather than in OPTION_SCHEMA. Absent -> the
    # global default (settings.recognize_concurrency); N=1 -> no pool (in-process
    # per-crop loop). Only recognizers use it (the pipeline fans a page's crops out).
    recognize_concurrency: dict[str, int] = field(default_factory=dict)
    # {engine_name: K} per-recognizer max concurrent images through the detect+recognize
    # half (the InferenceGate size). Like recognize_concurrency this is per-recognizer and
    # LOAD-TIME (the gate is rebuilt when the active recognizer or its K changes, not per
    # crop). Absent -> the global default (settings.gpu_concurrency); K=1 -> serial (today's
    # behavior). K>1 lets several images' crops fill the SHARED worker pool together
    # (cross-image overlap), so it only pays off with a pool (recognize_concurrency > 1).
    gpu_concurrency: dict[str, int] = field(default_factory=dict)
    # Active LLM system-prompt preset name (see app.prompts) + user-saved presets.
    prompt_active: str = "default"
    prompts: dict[str, str] = field(default_factory=dict)
    # Client behavior (delivered to the extension via the handshake): skip images
    # whose shorter side is under this many px (icons/banners). 0 = no filter.
    min_image_dim: int = settings.min_image_dim
    # Verbose (DEBUG) logging: per-detection/translation detail (see app.pipeline).
    # Seeded from SCANLATION_LOG_LEVEL (DEBUG -> on), toggled at runtime in /admin
    # (동작 tab) and re-applied to the scanlation logger without a restart.
    verbose_log: bool = settings.log_level.upper() == "DEBUG"
    # Max images translating concurrently off the GPU lock (bounds how many the
    # server sends to ollama at once). Seeded from SCANLATION_TRANSLATE_CONCURRENCY
    # (floor 1), edited in /admin (동작 tab) and applied at runtime by swapping
    # translate_sem (see AppState.set_client_config).
    # Default 1 is the safe floor: it never exceeds ollama's OLLAMA_NUM_PARALLEL
    # (whatever it is), so no request queues into a timeout. Even at 1 the translate
    # overlaps the next image's detect+recognize (it runs off the GPU lock). Parallel
    # GENERATION is opt-in: raise this AND OLLAMA_NUM_PARALLEL together (the lower of
    # the two = real parallelism; ollama can't be queried, so keeping them in sync is manual).
    translate_concurrency: int = settings.translate_concurrency
    # Idle-unload timer (MINUTES) for local torch engines (detector/recognizer): a
    # background sweep drops one from VRAM after it's gone this long without use, so
    # it doesn't hold the GPU between reading sessions. Translators are HTTP/out-of-
    # process and unaffected — ollama's own OLLAMA_KEEP_ALIVE governs those. Seeded
    # from SCANLATION_MODEL_IDLE_UNLOAD_MINUTES, edited in /admin (동작 tab). 0 = never.
    model_idle_unload_minutes: int = settings.model_idle_unload_minutes
    # GPU/torch build for plugin installs (the /admin 동작 tab). "cpu" (default) or
    # "gpu"; on "gpu" the vendor is auto-detected from device nodes at install time
    # (app.gpus.detect_gpu_vendor). torch is ONE build = ONE vendor, so this decides
    # which torch wheel a plugin install pulls — applied on the NEXT install. Seeded
    # from SCANLATION_TORCH_BACKEND/_VENDOR/_INDEX.
    torch_backend: str = settings.torch_backend
    # Force the vendor when BOTH AMD+NVIDIA are detected ("" = auto). "amd" | "nvidia".
    torch_vendor: str = settings.torch_vendor
    # Optional pip index URL override (blank = vendor default) — e.g. a specific
    # ROCm version whose prebuilt wheel matches the host (rocm6.2 vs 6.1, RDNA4…).
    torch_index: str = settings.torch_index


class AppState:
    """Process-wide selection + locks. One instance per process."""

    def __init__(self) -> None:
        self._path: Path = settings.data_dir / "state.json"
        self._lock = threading.Lock()
        self.selection = self._load()
        # Bounded-concurrency gate over the detect+recognize half (one device).
        # readers = inference (up to K=active recognizer's gpu_concurrency at once);
        # writers = lifecycle mutations (device/W change, idle-unload) drain all
        # permits. K=1 = single mutex. Translation (ollama) is a separate process and
        # runs off the gate, so one image's translate overlaps the next's recognize.
        self.gpu_gate = InferenceGate(self.resolve_gpu_concurrency(self.selection.recognizer))
        # Serializes detect(): the detector is a SHARED in-process torch model, so
        # concurrent readers must not forward through it at once (recognize fans out
        # to separate worker processes and is safe; detect stays serial — it's a small
        # slice of the half, so serializing it costs little overlap).
        self.detect_lock = threading.Lock()
        # Bound concurrent translations (they run off the GPU lock) so many
        # in-flight images don't overrun the ollama backend's parallel slots.
        # Seeded from the persisted selection; swapped at runtime by set_client_config.
        self.translate_sem = asyncio.Semaphore(self.selection.translate_concurrency)
        # md5/opts identity -> in-flight Future, so duplicate concurrent
        # requests for the same image attach to one computation.
        self.inflight: dict[tuple, asyncio.Future] = {}

    def _load(self) -> Selection:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return Selection(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return Selection()

    def save(self) -> None:
        with self._lock:
            settings.ensure_dirs()
            self._path.write_text(json.dumps(asdict(self.selection), ensure_ascii=False, indent=2), encoding="utf-8")

    # --- mutations (validated against the registry by the caller) ---
    def set_engines(self, detector: str | None, recognizer: str | None, translator: str | None) -> None:
        rec_changed = bool(recognizer) and recognizer != self.selection.recognizer
        if detector:
            self.selection.detector = detector
        if recognizer:
            self.selection.recognizer = recognizer
        if translator:
            self.selection.translator = translator
        self.save()
        if rec_changed:
            self.rebuild_gpu_gate()  # the new recognizer's K sizes the gate

    def set_languages(self, lang_src: str, lang_dst: str) -> None:
        self.selection.lang_src = lang_src
        self.selection.lang_dst = lang_dst
        self.save()

    def set_engine_device(self, engine_name: str, device: str | None) -> None:
        """Persist a per-engine compute-device override. Empty/None removes it, so
        the engine falls back to its DEFAULT_DEVICE. The caller must drop that
        engine's cached instance (registry.unload_one) so it reloads on the new
        device."""
        if device:
            self.selection.devices[engine_name] = device
        else:
            self.selection.devices.pop(engine_name, None)
        self.save()

    def resolve_device_for(self, engine_name: str) -> str | None:
        """The per-engine device override, or None to let the engine use its
        DEFAULT_DEVICE (there is no global device)."""
        return self.selection.devices.get(engine_name)

    def set_recognize_concurrency(self, engine_name: str, workers: int | None) -> None:
        """Persist a per-engine recognize worker-pool size. ``None`` removes the
        override (falls back to the global default); an explicit int is stored (incl.
        1, which forces 'no pool' for this engine even if the global default is
        higher). Floored at 1. The caller invalidates the pool so the next run
        rebuilds at the new size."""
        if workers is None:
            self.selection.recognize_concurrency.pop(engine_name, None)
        else:
            self.selection.recognize_concurrency[engine_name] = max(1, int(workers))
        self.save()

    def resolve_recognize_concurrency(self, engine_name: str) -> int:
        """The per-engine recognize worker-pool size, or the global default. Floor 1
        (1 = no pool). Read on every run to pick the in-process loop vs the pool."""
        return max(1, int(self.selection.recognize_concurrency.get(
            engine_name, settings.recognize_concurrency)))

    def set_gpu_concurrency(self, engine_name: str, k: int | None) -> None:
        """Persist a per-recognizer gate size (max concurrent images). ``None``
        removes the override (falls back to the global default); an explicit int is
        stored (incl. 1 = serial). Floored at 1. The caller rebuilds the gate when
        this is the active recognizer so the new size takes effect."""
        if k is None:
            self.selection.gpu_concurrency.pop(engine_name, None)
        else:
            self.selection.gpu_concurrency[engine_name] = max(1, int(k))
        self.save()

    def resolve_gpu_concurrency(self, engine_name: str) -> int:
        """The per-recognizer gate size, or the global default. Floor 1 (1 = serial)."""
        return max(1, int(self.selection.gpu_concurrency.get(
            engine_name, settings.gpu_concurrency)))

    def rebuild_gpu_gate(self) -> None:
        """Swap in a fresh gate sized for the ACTIVE recognizer's K. In-flight readers
        finish on the old gate; new requests use the new one (run_page reads
        state.gpu_gate each call) — the same runtime-swap pattern as translate_sem."""
        self.gpu_gate = InferenceGate(self.resolve_gpu_concurrency(self.selection.recognizer))

    def set_client_config(
        self, *, min_image_dim: int | None = None, verbose_log: bool | None = None,
        translate_concurrency: int | None = None, model_idle_unload_minutes: int | None = None,
        torch_backend: str | None = None, torch_vendor: str | None = None,
        torch_index: str | None = None,
    ) -> None:
        """Persist behavior settings (the /admin 동작 tab): the extension image
        filter, the verbose-log toggle, the concurrent-translation limit, the idle
        model-unload timer, and the GPU/torch backend for plugin installs. Verbose
        re-applies to the live logger and translate_concurrency swaps translate_sem
        (runtime, no restart); the idle timer is read live by the background sweep;
        the torch backend takes effect on the NEXT plugin install."""
        if min_image_dim is not None:
            self.selection.min_image_dim = max(0, int(min_image_dim))
        if verbose_log is not None:
            self.selection.verbose_log = bool(verbose_log)
            from .logconfig import apply_verbose
            apply_verbose(self.selection.verbose_log)
        if translate_concurrency is not None:
            self.selection.translate_concurrency = max(1, int(translate_concurrency))
            # New Semaphore instance: in-flight translates finish on the old one, new
            # requests use the new limit (run_page reads state.translate_sem each call).
            self.translate_sem = asyncio.Semaphore(self.selection.translate_concurrency)
        if model_idle_unload_minutes is not None:
            # Clamp is the single validation authority (>= 0; 0 = never unload). No
            # runtime re-apply — the idle sweep reads this fresh on each pass.
            self.selection.model_idle_unload_minutes = max(0, int(model_idle_unload_minutes))
        if torch_backend is not None:
            self.selection.torch_backend = torch_backend if torch_backend in ("cpu", "gpu") else "cpu"
        if torch_vendor is not None:
            self.selection.torch_vendor = torch_vendor if torch_vendor in ("", "amd", "nvidia") else ""
        if torch_index is not None:
            self.selection.torch_index = torch_index.strip()
        self.save()

    def set_options(self, engine_name: str, options: dict[str, Any]) -> None:
        """Persist per-engine option overrides (the admin 'engine options' form).

        A value of None or "" *removes* the override, so the field reverts to the
        engine's schema default (e.g. clearing the translator model unsets it).
        """
        cur = dict(self.selection.options.get(engine_name, {}))
        for key, val in (options or {}).items():
            if val is None or val == "":
                cur.pop(key, None)
            else:
                cur[key] = val
        if cur:
            self.selection.options[engine_name] = cur
        else:
            self.selection.options.pop(engine_name, None)
        self.save()

    def options_for(self, engine_name: str, request_options: dict | None) -> dict:
        """Merge persisted overrides with this request's options (request wins)."""
        merged = dict(self.selection.options.get(engine_name, {}))
        if request_options:
            merged.update(request_options.get(engine_name, {}) or {})
        return merged

    # --- LLM system prompt (shared by all LLM translators) ---
    def active_system_prompt(self) -> str:
        from .prompts import resolve_prompt

        return resolve_prompt(self.selection.prompt_active, self.selection.prompts)

    def translator_options(self, engine_name: str, request_options: dict | None) -> dict:
        """options_for + the active system prompt injected (unless overridden)."""
        opts = self.options_for(engine_name, request_options)
        opts.setdefault("system_prompt", self.active_system_prompt())
        return opts

    def save_prompt(self, name: str, text: str) -> None:
        """Upsert a user prompt preset and make it active."""
        self.selection.prompts[name] = text
        self.selection.prompt_active = name
        self.save()

    def select_prompt(self, name: str) -> None:
        self.selection.prompt_active = name
        self.save()

    def delete_prompt(self, name: str) -> None:
        self.selection.prompts.pop(name, None)
        if self.selection.prompt_active == name:
            self.selection.prompt_active = "default"
        self.save()


state = AppState()
