"""Auditable I/O helpers; no credentials and no provider-side mutations."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def progress(root, phase, cursor, **extra):
    record = dict(phase=phase, cursor=cursor, observed_at=time.time(), **extra)
    atomic_json(Path(root) / 'progress.json', record)
    print(json.dumps(record, allow_nan=False), flush=True)


def shared_root(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('experiment root must not be a symlink')
    result = path.resolve()
    # Large inputs and outputs must stay under one shared directory (default /workspace, a mounted
    # volume in the original runs); H3_SHARED_ROOT overrides it and skips the mount check.
    shared = Path(os.environ.get('H3_SHARED_ROOT', '/workspace'))
    if not result.is_relative_to(shared) or result == shared:
        raise ValueError('large inputs and outputs must remain in a shared experiment directory')
    if 'H3_SHARED_ROOT' not in os.environ and not shared.is_mount():
        raise ValueError('/workspace must be a mounted shared volume')
    return result


def source_hashes(directory):
    return {p.name: digest(p) for p in sorted(Path(directory).glob('*.py'))}
