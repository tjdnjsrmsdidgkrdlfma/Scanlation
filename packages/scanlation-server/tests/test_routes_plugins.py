"""routes/plugins.py + plugins_install unit tests — catalog and pip install flow."""
from __future__ import annotations

from tests.helpers import client, run


def test_catalog_lists_engines():
    from app.plugins_install import catalog

    c = catalog()
    for name in ("rtdetr", "mangaocr", "ollama", "llamacpp"):
        assert name in c, name
    assert "detector" in c["rtdetr"].roles
    assert "recognizer" in c["mangaocr"].roles
    assert "translator" in c["ollama"].roles
    assert c["rtdetr"].package == "scanlation-rtdetr"


def test_install_package_builds_pip_git_command():
    """Default install shells out to `pip install --upgrade --target=<vol> <sdk
    git+> <engine git+>` — no engine code is baked in; it's fetched from the repo.
    --upgrade forces the co-installed sdk to refresh. (Verified without actually
    installing.)"""
    import os

    from app import plugins_install as pi

    entry = pi.catalog()["ollama"]
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
    test_install_plugins,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_plugins"))
