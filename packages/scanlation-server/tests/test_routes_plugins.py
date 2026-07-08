"""routes/plugins.py + plugins_install unit tests — catalog and pip install flow."""
from __future__ import annotations

from tests.helpers import client, run


def test_catalog_lists_engines():
    from app.catalog import catalog

    c = catalog()
    for name in ("comic-text-and-bubble-detector", "manga-ocr", "Ollama", "llama.cpp"):
        assert name in c, name
    assert "detector" in c["comic-text-and-bubble-detector"].roles
    assert "recognizer" in c["manga-ocr"].roles
    assert "translator" in c["Ollama"].roles
    assert c["comic-text-and-bubble-detector"].package == "scanlation-comic-text-and-bubble-detector"


def test_install_package_builds_pip_git_command():
    """Default install shells out to `pip install --upgrade --target=<vol> <sdk
    git+> <engine git+>` — no engine code is baked in; it's fetched from the repo.
    --upgrade forces the co-installed sdk to refresh. (Verified without actually
    installing.)"""
    import os

    from app import plugins_install as pi
    from app.catalog import catalog

    entry = catalog()["Ollama"]
    recorded = {}

    class _Ok:
        returncode = 0
        stderr = ""
        stdout = ""

    orig_run = pi.subprocess.run
    orig_src = os.environ.pop("SCANLATION_ENGINES_SRC", None)  # force git mode
    pi.subprocess.run = lambda cmd, **kw: (recorded.__setitem__("cmd", cmd), _Ok())[1]
    try:
        pi.install_package(entry)
    finally:
        pi.subprocess.run = orig_run
        if orig_src is not None:
            os.environ["SCANLATION_ENGINES_SRC"] = orig_src

    cmd = recorded["cmd"]
    assert cmd[1:6] == ["-m", "pip", "install", "--upgrade", "--target"]
    assert str(pi.plugins_dir()) in cmd
    joined = " ".join(cmd)
    assert "git+" in joined
    assert "#subdirectory=packages/scanlation-ollama" in joined  # the engine
    assert "#subdirectory=packages/scanlation-sdk" in joined     # co-installed sdk


def test_torch_pip_args_by_backend_and_vendor():
    """_pip_cmd splices the torch index by /admin backend + auto-detected vendor:
    cpu -> cpu wheel; gpu+amd -> rocm; gpu+nvidia -> plain PyPI ([]); both/none ->
    cpu fallback; torch_index overrides; torch_vendor forces vendor on 'both'."""
    from app import gpus
    from app import plugins_install as pi
    from app.catalog import catalog
    from app.state import state

    entry = catalog()["manga-ocr"]   # torch=True engine
    sel = state.selection
    saved = (sel.torch_backend, sel.torch_vendor, sel.torch_index, gpus.detect_gpu_vendor)
    try:
        def cmd(backend, detect, vendor="", index=""):
            sel.torch_backend, sel.torch_vendor, sel.torch_index = backend, vendor, index
            gpus.detect_gpu_vendor = lambda: detect
            return " ".join(pi._pip_cmd(entry))

        assert "whl/cpu" in cmd("cpu", detect="amd")                        # cpu backend -> cpu wheel
        c = cmd("gpu", detect="amd")
        assert "whl/rocm" in c and "pypi.org/simple" in c                   # gpu + amd -> rocm
        assert "download.pytorch.org" not in cmd("gpu", detect="nvidia")    # gpu + nvidia -> plain PyPI
        assert "whl/cpu" in cmd("gpu", detect="both")                       # both unresolved -> cpu fallback
        assert "whl/rocm6.5" in cmd("gpu", detect="amd", index="https://x/whl/rocm6.5")  # index override
        assert "whl/rocm" in cmd("gpu", detect="both", vendor="amd")        # torch_vendor forces amd
    finally:
        sel.torch_backend, sel.torch_vendor, sel.torch_index, gpus.detect_gpu_vendor = saved


def test_install_plugins():
    c = client()
    # dummy has no assets -> install is a no-op success
    r = c.post("/install_plugins/", json={"plugins": {"dummy": True}})
    assert r.status_code == 200 and r.json()["status"] == "success"
    # unknown engine -> 502
    r2 = c.post("/install_plugins/", json={"plugins": {"nope": True}})
    assert r2.status_code == 502


# --- streaming install ------------------------------------------------------
def test_line_tee_splits_progress_from_log():
    """\\r-terminated segments are `progress` (a bar the UI overwrites), \\n ones are
    `log`. Blank segments are dropped; a trailing partial line comes out on flush."""
    from app.plugins_install import _LineTee

    seen: list[tuple] = []
    tee = _LineTee(seen.append)
    assert tee.isatty()                       # keeps tqdm/hf_hub bars enabled
    assert tee.write("done\n") == 5           # write() returns chars consumed
    tee.write("  50%\r  60%\r")               # bars, stripped
    tee.write("\n\n")                         # blank segments -> nothing
    tee.write("trailing, no delimiter")
    assert seen == [("log", "done"), ("progress", "50%"), ("progress", "60%")]
    tee.flush()
    assert seen[-1] == ("log", "trailing, no delimiter")


def test_begin_install_claims_a_name_once():
    """A second click while an install runs must not start a duplicate pip into the
    same --target (that corrupts it), and /get_settings/ must see what's running."""
    from app import plugins_install as pi

    assert pi._begin_install("zz-probe") is True
    try:
        assert pi._begin_install("zz-probe") is False   # already claimed
        assert "zz-probe" in pi.installing_names()
    finally:
        with pi._installing_lock:
            pi._installing.discard("zz-probe")
    assert "zz-probe" not in pi.installing_names()


class _FakePopen:
    """Stands in for pip: yields the given lines, then exits with `code`."""

    def __init__(self, lines, code):
        self.stdout = iter(lines)
        self._code = code
        self.returncode = None

    def wait(self):
        self.returncode = self._code


def _with_fake_popen(lines, code, fn):
    from app import plugins_install as pi

    orig = pi.subprocess.Popen
    pi.subprocess.Popen = lambda cmd, **kw: _FakePopen(lines, code)
    try:
        return fn()
    finally:
        pi.subprocess.Popen = orig


def test_stream_pip_forwards_every_line():
    from app import plugins_install as pi
    from app.catalog import catalog

    entry = catalog()["Ollama"]  # torch=False -> no state/gpu lookup in the pip cmd
    seen: list[tuple] = []
    _with_fake_popen(["Collecting x\n", "\n", "Successfully installed\n"], 0,
                     lambda: pi._stream_pip(entry, seen.append))
    assert seen == [("log", "Collecting x"), ("log", "Successfully installed")]  # blanks dropped


def test_stream_pip_raises_with_the_failure_tail():
    from app import plugins_install as pi
    from app.catalog import catalog

    entry = catalog()["Ollama"]
    lines = [f"line {i}\n" for i in range(20)] + ["ERROR: boom\n"]
    try:
        _with_fake_popen(lines, 1, lambda: pi._stream_pip(entry, lambda _: None))
    except RuntimeError as exc:
        msg = str(exc)
        assert "pip install scanlation-ollama failed" in msg
        assert "ERROR: boom" in msg and "line 15" in msg  # last 6 lines survive
        assert "line 5" not in msg                        # earlier ones are dropped
    else:
        raise AssertionError("a non-zero pip exit must raise")


def test_install_plugin_events_present_engine_yields_done():
    """dummy is already discoverable and has no weights -> package present, no pip,
    no weights phase, one `done` event carrying the same dict install_plugin returns."""
    from app import plugins_install as pi

    client()  # registers the fakes into the live registry
    events = list(pi.install_plugin_events("dummy"))
    assert events == [{"event": "done", "result": {"package": "present", "weights": "installed"}}]
    assert pi.installing_names() == []  # the claim is released in `finally`


def test_install_plugin_events_unknown_engine_yields_error():
    from app import plugins_install as pi

    events = list(pi.install_plugin_events("nope"))
    assert events == [{"event": "error", "message": "unknown engine: nope"}]
    assert pi.installing_names() == []


def test_install_plugin_stream_route_emits_ndjson():
    import json

    c = client()
    r = c.post("/install_plugin_stream/", json={"name": "dummy"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.splitlines() if line]
    assert events[-1]["event"] == "done"
    assert events[-1]["result"]["package"] == "present"


TESTS = [
    test_catalog_lists_engines,
    test_install_package_builds_pip_git_command,
    test_torch_pip_args_by_backend_and_vendor,
    test_install_plugins,
    test_line_tee_splits_progress_from_log,
    test_begin_install_claims_a_name_once,
    test_stream_pip_forwards_every_line,
    test_stream_pip_raises_with_the_failure_tail,
    test_install_plugin_events_present_engine_yields_done,
    test_install_plugin_events_unknown_engine_yields_error,
    test_install_plugin_stream_route_emits_ndjson,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_plugins"))
