"""File-system helpers for paths that the plain Windows API cannot handle.

Windows refuses paths over ~260 characters unless they carry the extended-length prefix, and
worst of all `Path.is_file()` then just returns False: a document whose file name is long (a
web page saved as PDF can have a 200-character name) silently vanishes from a scan. Every
place that lists or reads raw files goes through `long_path`.
"""

import os
from pathlib import Path

_EXTENDED = "\\\\?\\"
_EXTENDED_UNC = "\\\\?\\UNC\\"


def long_path(path: Path) -> Path:
    """The extended-length form of `path` on Windows; unchanged elsewhere. Always absolute."""
    if os.name != "nt":
        return path
    resolved = str(path.resolve())
    if resolved.startswith(_EXTENDED):
        return Path(resolved)
    if resolved.startswith("\\\\"):  # a network path: \\server\share -> \\?\UNC\server\share
        return Path(_EXTENDED_UNC + resolved[2:])
    return Path(_EXTENDED + resolved)
