# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Private file snapshots with filesystem copy-on-write when available."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys


def _try_clone(source: str, destination: str) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        clone = ctypes.CDLL(None).clonefile
    except AttributeError:
        return False
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if b"\0" in source_bytes or b"\0" in destination_bytes:
        return False
    clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    return clone(source_bytes, destination_bytes, 0) == 0


def copy_file(source: str, destination: str) -> None:
    """Copy contents, using a private copy-on-write clone for new files."""
    if (
        os.path.isfile(source)
        and not os.path.lexists(destination)
        and _try_clone(source, destination)
    ):
        return
    shutil.copyfile(source, destination)
