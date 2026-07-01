"""Test package.

Isolates the data dir BEFORE any ``app.*`` import so tests never touch real
cache/state files. Importing the package (which `python -m tests` and
`python -m tests.<mod>` both do first) runs this, so no test module has to
worry about import ordering.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("SCANLATION_BASE_DIR", tempfile.mkdtemp(prefix="scanlation-test-"))
