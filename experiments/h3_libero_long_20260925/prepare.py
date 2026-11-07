"""Pinned input acquisition, post-action audit and frozen-feature caching."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import torch
from common import atomic_json, digest, progress, shared_root, source_hashes
from encoders import visual_model, encode_images, encode_instructions, VISUAL_REPO, VISUAL_REV, TEXT_REPO, TEXT_REV


def audit_demo(demo):
    n = len(demo['actions'])
    if n < 9 or demo['actions'].shape != (n, 7):
        raise ValueError('invalid action trajectory')
    for key, dim in [('joint_states', 7), ('gripper_states', 2)]:
        if demo['obs/' + key].shape != (n, dim):
            raise ValueError('invalid proprioception')
    world = ET.fromstring(demo.attrs['model_file']).find('worldbody')
    joints = [node for node in world.iter() if node.tag in ('joint', 'freejoint')]
    if [(x.attrib.get('name'), x.attrib.get('type', 'hinge')) for x in joints[:7]] != [(f'robot0_joint{i}', 'hinge') for i in range(1, 8)]:
        raise ValueError('unverified MuJoCo joint ordering')
    states = np.asarray(demo['states'])
    observed = np.asarray(demo['obs/joint_states'])
    err = float(np.abs(observed[:-1] - states[1:, 1:8]).max())
    same = float(np.abs(observed[:-1] - states[:-1, 1:8]).mean())
    if not np.isfinite(states).all() or not np.isfinite(observed).all() or err > 1e-7 or same <= 1e-6:
        raise ValueError(f'post-action timing not established: {err}, {same}')
    for key in ('agentview_rgb', 'eye_in_hand_rgb'):
        if demo['obs/' + key].shape != (n, 128, 128, 3) or demo['obs/' + key].dtype != np.uint8:
            raise ValueError('unexpected image format')
    return dict(length=n, next_row_max=err, same_row_mae=same)


def download(root, manifest, quota_gb):
    needed = sum(x['bytes'] for x in manifest['files'])
    used = int(subprocess.check_output(['du', '-s', '-B1', '/workspace'], text=True).split()[0])
    reserve = 20_000_000_000
    if used + needed + reserve > quota_gb * 1_000_000_000 or shutil.disk_usage(root).free < needed + reserve:
        raise RuntimeError(f'not enough shared capacity: used={used}, need={needed + reserve}; no deletion permitted')
    for i, item in enumerate(manifest['files']):
        relative = Path(item['path'])
        if relative.parent != Path('libero_10') or relative.name != str(relative.name):
            raise ValueError('unexpected manifest path')
        target = root / 'data' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError('input target is a symlink')
        if target.exists():
            if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
                raise ValueError('existing input does not match the fixed source')
            continue
        partial = target.with_suffix('.partial')
        if partial.exists():
            raise FileExistsError('partial input retained; inspect before retrying')
        url = f'https://huggingface.co/datasets/{manifest["dataset_repository"]}/resolve/{manifest["revision"]}/{item["path"]}'
        progress(root, 'download', i * 10**12, file=item['path'], bytes=0)
        checksum, count, last = hashlib.sha256(), 0, time.monotonic()
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as stream:
            while block := response.read(8 * 1024**2):
                stream.write(block); checksum.update(block); count += len(block)
                if time.monotonic() - last > 5:
                    progress(root, 'download', i * 10**12 + count, file=item['path'], bytes=count)
                    last = time.monotonic()
            stream.flush(); os.fsync(stream.fileno())
        if count != item['bytes'] or checksum.hexdigest() != item['sha256']:
            raise ValueError('download size/hash mismatch')
        partial.replace(target)
        progress(root, 'download', (i + 1) * 10**12, completed_files=i + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--quota-gb', type=int, required=True)
    parser.add_argument('--image-batch', type=int, default=32)
    args = parser.parse_args()
    root = shared_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'prepared.json').exists():
        raise FileExistsError('preparation already completed; verify and reuse explicitly')
    manifest_path = Path(__file__).with_name('input_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if len(manifest['files']) != 10:
        raise ValueError('expected ten tasks')
    download(root, manifest, args.quota_gb)
    progress(root, 'audit', 0)
    tasks, records = [], []
    for item in manifest['files']:
        path = root / 'data' / item['path']
        with h5py.File(path, 'r') as h:
            if set(h['data']) != {f'demo_{i}' for i in range(50)} or h['data'].attrs['macros_image_convention'] != 'opengl':
                raise ValueError('unexpected demos/image convention')
            info = json.loads(h['data'].attrs['problem_info'])
            instruction = info['language_instruction']
            tasks.append(dict(file=path.name, instruction=instruction))
            for d in range(50):
                records.append(dict(file=path.name, demo=d, **audit_demo(h[f'data/demo_{d}'])))
        progress(root, 'audit', len(records))
    atomic_json(root / 'temporal_audit.json', dict(status='post_action_500_verified', records=records))
    cache = root / 'models'
    cache.mkdir(exist_ok=True)
    progress(root, 'text_features', 0)
    texts = encode_instructions([x['instruction'] for x in tasks], cache)
    atomic_json(root / 'language.json', dict(repository=TEXT_REPO, revision=TEXT_REV,
                tasks=[dict(**t, embedding=e.tolist()) for t, e in zip(tasks, texts)]))
    progress(root, 'visual_model_load', 0)
    model = visual_model(cache)
    features = root / 'features'
    features.mkdir(exist_ok=True)
    total, files = 0, []
    started = time.monotonic()
    for item in manifest['files']:
        path = root / 'data' / item['path']
        output = features / path.name
        if output.exists() or output.with_suffix('.partial').exists():
            raise FileExistsError('feature output already exists; no unverified overwrite')
        with h5py.File(path, 'r') as h, h5py.File(output.with_suffix('.partial'), 'x') as out:
            out.attrs['source_sha256'] = item['sha256']
            out.attrs['visual_revision'] = VISUAL_REV
            for d in range(50):
                src = h[f'data/demo_{d}']; n = len(src['actions'])
                dst = out.create_group(f'demo_{d}')
                dst.create_dataset('actions', data=src['actions'][:].astype('float32'))
                dst.create_dataset('states', data=np.concatenate([src['obs/joint_states'][:], src['obs/gripper_states'][:]], -1).astype('float32'))
                z = dst.create_dataset('visual', shape=(n, 2048), dtype='float32')
                for offset in range(0, n, args.image_batch):
                    stop = min(n, offset + args.image_batch)
                    parts = [encode_images(model, src[f'obs/{key}'][offset:stop]) for key in ('agentview_rgb', 'eye_in_hand_rgb')]
                    z[offset:stop] = np.concatenate(parts, -1)
                    total += stop - offset
                    progress(root, 'visual_features', total, file=path.name, demo=d, row=stop,
                             observations_per_second=total / (time.monotonic() - started))
                out.flush()
        output.with_suffix('.partial').replace(output)
        files.append(dict(file=output.name, sha256=digest(output)))
    report = dict(status='prepared', observation_timing='post_action', frames=total, files=files,
                  manifest_sha256=digest(manifest_path), language_sha256=digest(root/'language.json'),
                  temporal_audit_sha256=digest(root/'temporal_audit.json'), source_sha256=source_hashes(Path(__file__).parent),
                  visual_repository=VISUAL_REPO, visual_revision=VISUAL_REV, visual_dtype='bf16_autocast_fp32_pool',
                  static_frame_tubelet=2, predictor_used=False, feature_dim=2048,
                  device=torch.cuda.get_device_name(0))
    atomic_json(root / 'prepared.json', report)
    progress(root, 'prepared', total)


if __name__ == '__main__':
    main()
