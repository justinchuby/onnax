# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Snapshot ownership must survive writes and source removal on every filesystem."""

import pytest

from onnax import _file_snapshot


@pytest.mark.parametrize("fallback", [False, True])
def test_private_file_copy(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(_file_snapshot, "_try_clone", lambda *_: False)
    source, target = tmp_path / "source.data", tmp_path / "target.data"
    original = bytes(range(256)) * 4096
    source.write_bytes(original)
    _file_snapshot.copy_file(str(source), str(target))
    assert source.stat().st_ino != target.stat().st_ino
    assert target.stat().st_nlink == 1
    source.write_bytes(b"changed source")
    assert target.read_bytes() == original
    target.write_bytes(b"changed target")
    assert source.read_bytes() == b"changed source"
    source.unlink()
    assert target.read_bytes() == b"changed target"


def test_existing_destination_and_source_symlink(tmp_path):
    source, target, link = (tmp_path / name for name in ("source", "target", "link"))
    source.write_bytes(b"original")
    target.write_bytes(b"old contents")
    link.symlink_to(source)
    _file_snapshot.copy_file(str(link), str(target))
    assert not target.is_symlink() and target.read_bytes() == b"original"
    source.write_bytes(b"replacement")
    assert target.read_bytes() == b"original"
    fresh = tmp_path / "fresh"
    _file_snapshot.copy_file(str(link), str(fresh))
    assert not fresh.is_symlink() and fresh.read_bytes() == b"replacement"


def test_missing_source_does_not_create_destination(tmp_path):
    target = tmp_path / "target"
    with pytest.raises(FileNotFoundError):
        _file_snapshot.copy_file(str(tmp_path / "missing"), str(target))
    assert not target.exists()


def test_destination_with_null_byte_is_rejected(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError):
        _file_snapshot.copy_file(str(source), str(target) + "\0suffix")
    assert not target.exists()
