"""Task-balanced causal windows over immutable, precomputed encoder features."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from common import digest, progress
from temporal_contract import causal_window, valid_anchors


class Store:
    def __init__(self, root, *, verify=True, report=None):
        self.root = Path(root)
        self.prepared = json.loads((self.root / 'prepared.json').read_text())
        if self.prepared['observation_timing'] != 'post_action':
            raise ValueError('input timing was not audited')
        self.language = json.loads((self.root / 'language.json').read_text())
        if digest(self.root / 'language.json') != self.prepared['language_sha256']:
            raise ValueError('language cache hash mismatch')
        self.episodes, self.index = {}, {split: [[] for _ in range(10)] for split in ('train', 'validation')}
        self.boundaries = {split: [[] for _ in range(10)] for split in ('train', 'validation')}
        self.text = np.asarray([t['embedding'] for t in self.language['tasks']], dtype='float32')
        if self.text.shape != (10, 768) or not np.isfinite(self.text).all():
            raise ValueError('invalid text features')
        sums, squares, count = np.zeros(2048), np.zeros(2048), 0
        amin, amax = np.full(7, np.inf), np.full(7, -np.inf)
        smin, smax = np.full(9, np.inf), np.full(9, -np.inf)
        for task, item in enumerate(self.prepared['files']):
            if item['file'] != self.language['tasks'][task]['file'] or Path(item['file']).name != item['file']:
                raise ValueError('task ordering mismatch')
            path = self.root / 'features' / item['file']
            if verify and digest(path) != item['sha256']:
                raise ValueError('feature hash mismatch')
            with h5py.File(path, 'r') as h:
                if set(h) != {f'demo_{i}' for i in range(50)}:
                    raise ValueError('incomplete feature cache')
                for demo in range(50):
                    arrays = {k: np.asarray(h[f'demo_{demo}/{k}'], dtype='float32') for k in ('visual', 'actions', 'states')}
                    n = len(arrays['visual'])
                    if any(arrays[k].shape != (n, d) or not np.isfinite(arrays[k]).all()
                           for k, d in [('visual', 2048), ('actions', 7), ('states', 9)]):
                        raise ValueError('invalid feature record')
                    self.episodes[task, demo] = arrays
                    split = 'train' if demo < 45 else 'validation'
                    for anchor in valid_anchors(n):
                        self.index[split][task].append((task, demo, anchor))
                        if anchor % 16 == 0:
                            self.boundaries[split][task].append((task, demo, anchor))
                    if demo < 45:
                        z = arrays['visual'].astype('float64')
                        sums += z.sum(0); squares += (z*z).sum(0); count += n
                        amin = np.minimum(amin, arrays['actions'].min(0)); amax = np.maximum(amax, arrays['actions'].max(0))
                        smin = np.minimum(smin, arrays['states'].min(0)); smax = np.maximum(smax, arrays['states'].max(0))
            if report:
                progress(report, 'load_features', task + 1)
        mean = sums / count
        variance = squares / count - mean*mean
        if np.any(variance <= 0) or not np.isfinite(variance).all():
            raise ValueError('nonpositive training feature variance')
        self.normalization = dict(visual_mean=mean.tolist(), visual_std=np.sqrt(variance).tolist(),
                                  action_min=amin.tolist(), action_max=amax.tolist(),
                                  state_min=smin.tolist(), state_max=smax.tolist(), training_frames=count)
        for arrays in self.episodes.values():
            arrays['visual'] = ((arrays['visual'] - mean) / np.sqrt(variance)).astype('float32')
            arrays['actions'] = limits(arrays['actions'], amin, amax).astype('float32')
            arrays['states'] = limits(arrays['states'], smin, smax).astype('float32')
        if any(not x for table in (self.index, self.boundaries) for split in table.values() for x in split):
            raise ValueError('empty task/split')

    def sample(self, rng, size, split='train', upper_only=False):
        table = self.boundaries if upper_only else self.index
        rows = []
        for task in rng.integers(10, size=size):
            items = table[split][int(task)]
            rows.append(self.row(items[int(rng.integers(len(items)))]))
        return {k: torch.from_numpy(np.stack([x[k] for x in rows])) for k in rows[0]}

    def row(self, index):
        task, demo, anchor = index
        e = self.episodes[task, demo]
        w = causal_window(len(e['visual']), anchor, observation_timing='post_action')
        acts = e['actions'][list(w.upper_actions)] if w.upper_action_mask else np.zeros((16, 7), 'float32')
        return dict(lower_visual=e['visual'][list(w.lower_observations)],
                    lower_states=e['states'][list(w.lower_observations)],
                    upper_visual=e['visual'][list(w.upper_observations)],
                    upper_actions=acts, upper_mask=np.array([w.upper_action_mask], 'float32'),
                    remaining=np.ones(1, 'float32'), language=self.text[task],
                    upper_teacher=e['visual'][w.upper_teacher],
                    future_actions=e['actions'][list(w.future_actions)])


def limits(value, minimum, maximum):
    minimum, maximum = np.asarray(minimum), np.asarray(maximum)
    scale = maximum - minimum
    scale = np.where(scale < 1e-8, 1., scale)
    return 2 * (value - minimum) / scale - 1


def inverse_actions(value, normalization):
    lo, hi = np.asarray(normalization['action_min']), np.asarray(normalization['action_max'])
    scale = np.where(hi-lo < 1e-8, 1., hi-lo)
    return np.clip((value + 1) * scale / 2 + lo, -1, 1).astype('float32')
