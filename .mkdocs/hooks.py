"""MkDocs hook: expose ``src/`` under its installed name ``sigconfide``.

pyproject.toml maps ``src`` to the ``sigconfide`` package, so the source tree
has no directory with that name for mkdocstrings (griffe) to find.
"""

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_LINK_DIR = Path(__file__).parent / "pkg"


def on_startup(**kwargs):
    _LINK_DIR.mkdir(exist_ok=True)
    link = _LINK_DIR / "sigconfide"
    if not link.exists():
        link.symlink_to(_ROOT / "src", target_is_directory=True)
