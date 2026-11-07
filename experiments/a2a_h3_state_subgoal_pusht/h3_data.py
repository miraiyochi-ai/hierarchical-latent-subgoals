#!/usr/bin/env python3
"""H3 PushT sample enumeration and leak-free HDF5 window loading."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch import Tensor

H1_DIR = Path(__file__).resolve().parents[1] / "a2a_h1_clockwork_pusht_gate"
if str(H1_DIR) not in sys.path:
    sys.path.insert(0, str(H1_DIR))
from h1_data import ACTION_MAXIMUM, ACTION_MINIMUM, CyclicSpecSampler, EpisodeTable, Normalizers, load_split_ids  # noqa: E402

HORIZONS = (25, 50, 75)
L1_HISTORY = 8
L1_CHUNK = 8
L2_CLOCK = 16
L2_WINDOW = 16
S2_DIM = 192
FULL_BATCH_SIZE = 128

def normalize_action(value: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Apply H1's fixed action_offset=0/action_scale=1 transformation."""
    return value.float() if isinstance(value, Tensor) else np.asarray(value,dtype=np.float32)


@dataclass(frozen=True, order=True)
class Layer2Spec:
    episode_idx: int
    task_start_step: int
    d: int
    boundary: int

    @property
    def goal_step(self) -> int:
        return self.task_start_step + self.d

    @property
    def teacher_step(self) -> int:
        return self.task_start_step + min(self.boundary + L2_CLOCK, self.d)

    @property
    def sample_id(self) -> str:
        return f"l2:{self.episode_idx}:{self.task_start_step}:{self.d}:{self.boundary}"


@dataclass(frozen=True, order=True)
class Layer1Spec:
    episode_idx: int
    task_start_step: int
    d: int
    u: int

    @property
    def boundary(self) -> int:
        return L2_CLOCK * (self.u // L2_CLOCK)

    @property
    def s2_id(self) -> str:
        return f"l2:{self.episode_idx}:{self.task_start_step}:{self.d}:{self.boundary}"

    @property
    def sample_id(self) -> str:
        return f"l1:{self.episode_idx}:{self.task_start_step}:{self.d}:{self.u}"


@dataclass(frozen=True)
class S2Normalizer:
    mu: np.ndarray
    sigma: np.ndarray
    count: int

    @classmethod
    def load(cls, path: Path) -> "S2Normalizer":
        with np.load(path, allow_pickle=False) as values:
            mu = np.asarray(values["mu"], dtype=np.float64)
            sigma = np.asarray(values["sigma"], dtype=np.float64)
            count = int(np.asarray(values["count"]).item())
        if mu.shape != (S2_DIM,) or sigma.shape != (S2_DIM,):
            raise ValueError("h3_input_blocked: s2 normalizer must have 192 dimensions")
        if count <= 0 or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("h3_input_blocked: invalid s2 normalizer")
        return cls(mu, sigma, count)

    def apply(self, latent: np.ndarray | Tensor) -> np.ndarray | Tensor:
        if isinstance(latent, Tensor):
            mu = torch.as_tensor(self.mu, dtype=latent.dtype, device=latent.device)
            sigma = torch.as_tensor(self.sigma, dtype=latent.dtype, device=latent.device)
            return (latent - mu) / sigma
        return ((np.asarray(latent, dtype=np.float64) - self.mu) / self.sigma).astype(np.float32)

    def invert(self, latent: np.ndarray | Tensor) -> np.ndarray | Tensor:
        if isinstance(latent, Tensor):
            return latent * torch.as_tensor(self.sigma, dtype=latent.dtype, device=latent.device) + torch.as_tensor(self.mu, dtype=latent.dtype, device=latent.device)
        return (np.asarray(latent) * self.sigma + self.mu).astype(np.float32)


def _valid_task_starts(length: int, d: int) -> range:
    # goal_step is an observed row, hence strict length-d.
    return range(max(0, length - d))


def enumerate_layer2_specs(table: EpisodeTable, episode_ids: Sequence[int]) -> dict[tuple[int, int], list[Layer2Spec]]:
    pools = {(d, b): [] for d in HORIZONS for b in range(0, d, L2_CLOCK)}
    for episode_idx in map(int, episode_ids):
        length = int(table.lengths[episode_idx])
        for d in HORIZONS:
            for start in _valid_task_starts(length, d):
                for boundary in range(0, d, L2_CLOCK):
                    pools[(d, boundary)].append(Layer2Spec(episode_idx, start, d, boundary))
    return pools


def enumerate_layer1_specs(table: EpisodeTable, episode_ids: Sequence[int]) -> dict[int, list[Layer1Spec]]:
    pools = {d: [] for d in HORIZONS}
    for episode_idx in map(int, episode_ids):
        length = int(table.lengths[episode_idx])
        for d in HORIZONS:
            for start in _valid_task_starts(length, d):
                pools[d].extend(Layer1Spec(episode_idx, start, d, u) for u in range(0, d - L1_CHUNK + 1, L1_CHUNK))
    return pools


def balanced_horizon_counts(update: int, batch_size: int = FULL_BATCH_SIZE) -> dict[int, int]:
    base, remainder = divmod(batch_size, len(HORIZONS))
    counts = {d: base for d in HORIZONS}
    order = HORIZONS[update % 3 :] + HORIZONS[: update % 3]
    for d in order[:remainder]:
        counts[d] += 1
    return counts


class BoundarySampler:
    """Balance d first, then deterministically cycle over boundaries within d."""

    def __init__(self, pools: Mapping[tuple[int, int], Sequence[Layer2Spec]], seed: int) -> None:
        self.sampler = CyclicSpecSampler(pools, seed)
        self.positions = {d: 0 for d in HORIZONS}
        self.boundaries = {d: tuple(range(0, d, L2_CLOCK)) for d in HORIZONS}

    def take_update(self, update: int, batch_size: int = FULL_BATCH_SIZE) -> list[Layer2Spec]:
        result: list[Layer2Spec] = []
        for d, count in balanced_horizon_counts(update, batch_size).items():
            for _ in range(count):
                boundaries = self.boundaries[d]
                boundary = boundaries[self.positions[d] % len(boundaries)]
                self.positions[d] += 1
                result.extend(self.sampler.take((d, boundary), 1))
        return result

    def take(self, d: int, count: int) -> list[Layer2Spec]:
        result=[]
        for _ in range(count):
            boundaries=self.boundaries[d]; boundary=boundaries[self.positions[d]%len(boundaries)]; self.positions[d]+=1; result.extend(self.sampler.take((d,boundary),1))
        return result


def _rows(table: EpisodeTable, episode: int, steps: Iterable[int]) -> np.ndarray:
    return np.asarray([table.global_row(episode, int(step)) for step in steps], dtype=np.int64)


def layer2_reference_steps(spec: Layer2Spec) -> dict[str, tuple[int, ...] | int]:
    current = spec.task_start_step + spec.boundary
    if spec.boundary == 0:
        observations = (current,) * L2_WINDOW
        actions: tuple[int, ...] = ()
    else:
        observations = tuple(range(current - L2_WINDOW + 1, current + 1))
        actions = tuple(range(current - L2_WINDOW, current))
    return {
        "observations": observations,
        "actions": actions,
        "current": current,
        "teacher": spec.teacher_step,
        "goal": spec.goal_step,
    }


def validate_no_leak(layer1: Sequence[Layer1Spec], layer2: Sequence[Layer2Spec]) -> dict[str, int]:
    violations = {"goal_overrun": 0, "boundary_overrun": 0, "wrong_s2_id": 0, "episode_crossing": 0}
    known = {spec.sample_id for spec in layer2}
    for spec in layer2:
        refs = layer2_reference_steps(spec)
        if int(refs["teacher"]) > spec.goal_step:
            violations["goal_overrun"] += 1
        if any(step > spec.task_start_step + spec.boundary for step in refs["observations"]):
            violations["boundary_overrun"] += 1
        if any(step >= spec.task_start_step + spec.boundary for step in refs["actions"]):
            violations["boundary_overrun"] += 1
    for spec in layer1:
        if spec.s2_id not in known:
            violations["wrong_s2_id"] += 1
        if spec.u + L1_CHUNK > spec.d:
            violations["goal_overrun"] += 1
    return violations


class H3HDF5Reader:
    """Return uint8 images; official preprocessing is deliberately model-side."""

    def __init__(self, path: Path, table: EpisodeTable, normalizers: Normalizers) -> None:
        self.path, self.table, self.normalizers = Path(path), table, normalizers
        self._handle: h5py.File | None = None

    def open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close(); self._handle = None

    def __enter__(self) -> "H3HDF5Reader": self.open(); return self
    def __exit__(self, *_: object) -> None: self.close()

    def _read(self, name: str, rows: np.ndarray) -> np.ndarray:
        self.open()
        assert self._handle is not None
        unique, inverse = np.unique(rows, return_inverse=True)
        return np.asarray(self._handle[name][unique])[inverse]

    def _pixels(self, rows: np.ndarray) -> np.ndarray:
        value = self._read("pixels", rows)
        if value.ndim != 4:
            raise ValueError("h3_input_blocked: pixels must be rank four")
        if value.shape[-1] == 3: value = value.transpose(0, 3, 1, 2)
        if value.shape[1] != 3 or value.dtype != np.uint8:
            raise ValueError("h3_input_blocked: optimized image path requires CHW uint8")
        return value

    def layer2(self, spec: Layer2Spec) -> dict[str, Tensor]:
        refs = layer2_reference_steps(spec)
        obs = self._pixels(_rows(self.table, spec.episode_idx, refs["observations"]))
        current = self._pixels(_rows(self.table, spec.episode_idx, [refs["current"]]))[0]
        teacher = self._pixels(_rows(self.table, spec.episode_idx, [refs["teacher"]]))[0]
        goal = self._pixels(_rows(self.table, spec.episode_idx, [refs["goal"]]))[0]
        actions = np.zeros((L2_WINDOW, 2), np.float32)
        mask = 0.0
        if refs["actions"]:
            actions[:] = normalize_action(self._read("action", _rows(self.table, spec.episode_idx, refs["actions"])))
            mask = 1.0
        remaining = min(L2_CLOCK, spec.d - spec.boundary) / L2_CLOCK
        return {"observation_window": torch.from_numpy(obs), "current_image": torch.from_numpy(current), "teacher_image": torch.from_numpy(teacher), "goal_image": torch.from_numpy(goal), "action_history": torch.from_numpy(actions), "history_mask": torch.tensor([mask]), "remaining_horizon": torch.tensor([remaining], dtype=torch.float32), "episode_idx": torch.tensor(spec.episode_idx), "task_start_step": torch.tensor(spec.task_start_step), "d": torch.tensor(spec.d), "boundary": torch.tensor(spec.boundary), "teacher_step": torch.tensor(spec.teacher_step)}

    def layer1(self, spec: Layer1Spec) -> dict[str, Tensor]:
        step = spec.task_start_step + spec.u
        history = [max(spec.task_start_step, step - L1_HISTORY + 1 + i) for i in range(L1_HISTORY)]
        future = range(step, step + L1_CHUNK)
        head = self._pixels(_rows(self.table, spec.episode_idx, history))
        agent = self.normalizers.agent(self._read("proprio", _rows(self.table, spec.episode_idx, history))[:, :2])
        actions = np.asarray(self._read("action", _rows(self.table, spec.episode_idx, future)), np.float32)
        # The layer-two inputs are reconstructed at the preceding boundary, never at u.
        boundary_record = self.layer2(Layer2Spec(spec.episode_idx, spec.task_start_step, spec.d, spec.boundary))
        return {"head_cam": torch.from_numpy(head), "agent_pos": torch.from_numpy(agent), "future_actions": torch.from_numpy(actions), "episode_idx": torch.tensor(spec.episode_idx), "task_start_step": torch.tensor(spec.task_start_step), "d": torch.tensor(spec.d), "u": torch.tensor(spec.u), "boundary": torch.tensor(spec.boundary), "s2_id_hash": torch.tensor(int(hashlib.sha256(spec.s2_id.encode()).hexdigest()[:15], 16), dtype=torch.int64), **{key: value for key, value in boundary_record.items() if key in {"observation_window", "current_image", "goal_image", "action_history", "history_mask", "remaining_horizon"}}}


def collate(records: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not records: raise ValueError("cannot collate an empty batch")
    return {key: torch.stack([row[key] for row in records]) for key in records[0]}


def fixed_hash_order(specs: Sequence[object], prefix: str) -> list[object]:
    return sorted(specs, key=lambda item: hashlib.sha256(f"{prefix}:{item}".encode()).hexdigest())


def load_tasks(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
