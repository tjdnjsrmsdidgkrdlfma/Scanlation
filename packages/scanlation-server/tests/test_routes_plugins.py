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
    # unknown plugin -> 502
    r2 = c.post("/install_plugins/", json={"plugins": {"nope": True}})
    assert r2.status_code == 502


TESTS = [
    test_catalog_lists_engines,
    test_install_package_builds_pip_git_command,
    test_torch_pip_args_by_backend_and_vendor,
    test_install_plugins,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_plugins"))
