#!/usr/bin/env python3
"""H1 PushT windows, split loading, normalization, and fixed batch allocation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import hdf5plugin  # noqa: F401  # Registers the compression filters used by the fixed HDF5.
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


HORIZONS = (25, 50, 75)
HISTORY_STEPS = 8
ACTION_STEPS = 8
CLOCK_STEPS = 32
FULL_BATCH_SIZE = 128
ACTION_MINIMUM = np.asarray([-1.49470758, -1.46553111], dtype=np.float32)
ACTION_MAXIMUM = np.asarray([2.04266977, 1.79205656], dtype=np.float32)


@dataclass(frozen=True, order=True)
class Layer1Spec:
    episode_idx: int
    start_step: int
    d: int


@dataclass(frozen=True, order=True)
class Layer2Spec:
    episode_idx: int
    task_start_step: int
    d: int
    boundary: int


@dataclass(frozen=True)
class EpisodeTable:
    lengths: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_hdf5(cls, path: Path) -> "EpisodeTable":
        with h5py.File(path, "r", swmr=True) as handle:
            lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
            offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        expected = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
        if not np.array_equal(offsets, expected):
            raise ValueError("h1_data_blocked: episode offsets are inconsistent")
        return cls(lengths=lengths, offsets=offsets)

    def global_row(self, episode_idx: int, step: int) -> int:
        if episode_idx < 0 or episode_idx >= len(self.lengths):
            raise IndexError("episode_idx is outside the HDF5 table")
        if step < 0 or step >= int(self.lengths[episode_idx]):
            raise IndexError("step crosses an episode boundary")
        return int(self.offsets[episode_idx] + step)


@dataclass(frozen=True)
class Normalizers:
    agent_pos_mean: np.ndarray
    agent_pos_std: np.ndarray
    image_divisor: float = 255.0

    @classmethod
    def load(cls, path: Path) -> "Normalizers":
        with np.load(path, allow_pickle=False) as data:
            mean = np.asarray(data["agent_pos_mean"], dtype=np.float32)
            std = np.asarray(data["agent_pos_std"], dtype=np.float32)
            divisor = float(np.asarray(data["image_divisor"]).reshape(-1)[0])
        if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0) or divisor != 255.0:
            raise ValueError("h1_data_blocked: invalid normalizer contract")
        return cls(mean, std, divisor)

    def agent(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) - self.agent_pos_mean) / self.agent_pos_std

    def image(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) / self.image_divisor


def load_split_ids(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("h1_data_blocked: unsupported split schema")
    key_map = {
        "train": "train_episode_ids",
        "validation": "validation_episode_ids",
        "test": "test_episode_ids",
    }
    result = {name: np.asarray(payload[key], dtype=np.int64) for name, key in key_map.items()}
    sets = {name: set(values.tolist()) for name, values in result.items()}
    if len(result["validation"]) != 256 or len(result["train"]) < 512:
        raise ValueError("h1_data_blocked: split counts violate the preregistration")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("h1_data_blocked: split episode IDs overlap")
    return result


def enumerate_layer1_specs(table: EpisodeTable, episode_ids: Sequence[int]) -> dict[int, list[Layer1Spec]]:
    result: dict[int, list[Layer1Spec]] = {d: [] for d in HORIZONS}
    for episode_idx in map(int, episode_ids):
        length = int(table.lengths[episode_idx])
        for d in HORIZONS:
            result[d].extend(Layer1Spec(episode_idx, start, d) for start in range(max(0, length - d)))
    return result


def enumerate_layer2_specs(table: EpisodeTable, episode_ids: Sequence[int]) -> dict[tuple[int, int], list[Layer2Spec]]:
    result: dict[tuple[int, int], list[Layer2Spec]] = {(50, 32): [], (75, 32): [], (75, 64): []}
    for episode_idx in map(int, episode_ids):
        length = int(table.lengths[episode_idx])
        for d, boundary in result:
            # The target chunk ends no later than the registered goal.
            if boundary + ACTION_STEPS > d:
                continue
            for task_start in range(max(0, length - d)):
                result[(d, boundary)].append(Layer2Spec(episode_idx, task_start, d, boundary))
    return result


def layer1_batch_counts(update: int, batch_size: int = FULL_BATCH_SIZE) -> dict[int, int]:
    if batch_size == FULL_BATCH_SIZE:
        counts = {25: 42, 50: 42, 75: 42}
        order = HORIZONS[update % 3 :] + HORIZONS[: update % 3]
        counts[order[0]] += 1
        counts[order[1]] += 1
        return counts
    if batch_size == 16:
        base, remainder = divmod(batch_size, 3)
        counts = {d: base for d in HORIZONS}
        order = HORIZONS[update % 3 :] + HORIZONS[: update % 3]
        for d in order[:remainder]:
            counts[d] += 1
        return counts
    raise ValueError("only the registered full or Mac batch size is allowed")


def layer2_batch_counts(batch_size: int = FULL_BATCH_SIZE) -> dict[tuple[int, int], int]:
    if batch_size == FULL_BATCH_SIZE:
        return {(50, 32): 64, (75, 32): 32, (75, 64): 32}
    if batch_size == 16:
        return {(50, 32): 8, (75, 32): 4, (75, 64): 4}
    raise ValueError("only the registered full or Mac batch size is allowed")


class CyclicSpecSampler:
    """Deterministic per-seed shuffles with exact registered category counts."""

    def __init__(self, pools: Mapping[object, Sequence[object]], seed: int) -> None:
        self.pools = {key: list(value) for key, value in pools.items()}
        if any(not pool for pool in self.pools.values()):
            raise ValueError("h1_data_blocked: an allocation pool is empty")
        self.rng = np.random.default_rng(seed)
        self.orders = {key: self.rng.permutation(len(pool)) for key, pool in self.pools.items()}
        self.positions = {key: 0 for key in self.pools}

    def take(self, key: object, count: int) -> list[object]:
        selected: list[object] = []
        while len(selected) < count:
            position = self.positions[key]
            order = self.orders[key]
            available = min(count - len(selected), len(order) - position)
            selected.extend(self.pools[key][int(i)] for i in order[position : position + available])
            position += available
            if position == len(order):
                order = self.rng.permutation(len(self.pools[key]))
                self.orders[key] = order
                position = 0
            self.positions[key] = position
        return selected


def _rows(table: EpisodeTable, episode: int, steps: Iterable[int]) -> np.ndarray:
    return np.asarray([table.global_row(episode, int(step)) for step in steps], dtype=np.int64)


class H1HDF5Reader:
    """Read only the rows required for a requested window."""

    def __init__(
        self,
        path: Path,
        table: EpisodeTable,
        normalizers: Normalizers,
        *,
        image_size: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.table = table
        self.normalizers = normalizers
        if image_size not in {None, 64, 224}:
            raise ValueError("only the registered full or Mac image size is allowed")
        self.image_size = image_size
        self._handle: h5py.File | None = None

    def open(self) -> None:
        """Open this process's private read-only handle."""
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "H1HDF5Reader":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read(self, name: str, rows: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
        self.open()
        if self._handle is None:
            raise RuntimeError("HDF5 reader failed to open its process-local handle")
        values = np.asarray(self._handle[name][unique])
        return values[inverse]

    def _pixels(self, rows: np.ndarray) -> np.ndarray:
        values = self._read("pixels", rows)
        if values.ndim != 4:
            raise ValueError("h1_data_blocked: pixels are not rank four")
        if values.shape[-1] == 3:
            values = values.transpose(0, 3, 1, 2)
        if values.shape[1] != 3:
            raise ValueError("h1_data_blocked: pixels do not have three channels")
        values = self.normalizers.image(values)
        if self.image_size is not None and values.shape[-2:] != (self.image_size, self.image_size):
            values = torch.nn.functional.interpolate(
                torch.from_numpy(values),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).numpy()
        return values

    def layer1(self, spec: Layer1Spec) -> dict[str, Tensor]:
        length = int(self.table.lengths[spec.episode_idx])
        if spec.d not in HORIZONS or spec.start_step + spec.d >= length:
            raise IndexError("layer-one specification crosses the episode")
        first = 0
        history_steps = [max(first, spec.start_step - HISTORY_STEPS + 1 + i) for i in range(HISTORY_STEPS)]
        future_steps = range(spec.start_step, spec.start_step + ACTION_STEPS)
        history_rows = _rows(self.table, spec.episode_idx, history_steps)
        future_rows = _rows(self.table, spec.episode_idx, future_steps)
        goal_row = _rows(self.table, spec.episode_idx, [spec.start_step + spec.d])
        head = self._pixels(history_rows)
        goal_one = self._pixels(goal_row)[0]
        goal = np.repeat(goal_one[None], HISTORY_STEPS, axis=0)
        agent = self.normalizers.agent(self._read("proprio", history_rows)[:, :2])
        future = np.asarray(self._read("action", future_rows), dtype=np.float32)
        if not np.isfinite(future).all():
            raise ValueError("h1_data_blocked: non-finite action")
        slow = np.zeros((4, 8, 2), dtype=np.float32)
        mask = 0.0
        if spec.start_step >= CLOCK_STEPS:
            slow_rows = _rows(
                self.table, spec.episode_idx, range(spec.start_step - CLOCK_STEPS, spec.start_step)
            )
            slow = np.asarray(self._read("action", slow_rows), dtype=np.float32).reshape(4, 8, 2)
            mask = 1.0
        return {
            "head_cam": torch.from_numpy(head),
            "goal_cam": torch.from_numpy(goal),
            "agent_pos": torch.from_numpy(agent),
            "future_actions": torch.from_numpy(future),
            "slow_history_actions": torch.from_numpy(slow),
            "slow_state_mask": torch.tensor([mask], dtype=torch.float32),
            "d": torch.tensor(spec.d, dtype=torch.int64),
            "episode_idx": torch.tensor(spec.episode_idx, dtype=torch.int64),
            "start_step": torch.tensor(spec.start_step, dtype=torch.int64),
        }

    def layer2(self, spec: Layer2Spec) -> dict[str, Tensor]:
        if (spec.d, spec.boundary) not in {(50, 32), (75, 32), (75, 64)}:
            raise ValueError("unregistered layer-two boundary")
        end = spec.task_start_step + spec.boundary
        if end + ACTION_STEPS > spec.task_start_step + spec.d:
            raise IndexError("layer-two teacher extends beyond the goal")
        history_rows = _rows(self.table, spec.episode_idx, range(end - CLOCK_STEPS, end))
        target_rows = _rows(self.table, spec.episode_idx, range(end, end + ACTION_STEPS))
        goal_row = _rows(self.table, spec.episode_idx, [spec.task_start_step + spec.d])
        history = np.asarray(self._read("action", history_rows), dtype=np.float32).reshape(4, 8, 2)
        target = np.asarray(self._read("action", target_rows), dtype=np.float32)
        goal_one = self._pixels(goal_row)[0]
        goal = np.repeat(goal_one[None], HISTORY_STEPS, axis=0)
        return {
            "history_actions": torch.from_numpy(history),
            "target_actions": torch.from_numpy(target),
            "goal_cam": torch.from_numpy(goal),
            "d": torch.tensor(spec.d, dtype=torch.int64),
            "boundary": torch.tensor(spec.boundary, dtype=torch.int64),
        }


def collate(records: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot collate an empty batch")
    return {key: torch.stack([record[key] for record in records]) for key in records[0]}


def fixed_hash_order(specs: Sequence[Layer1Spec], prefix: str) -> list[Layer1Spec]:
    return sorted(
        specs,
        key=lambda spec: (
            hashlib.sha256(
                f"{prefix}:{spec.episode_idx}:{spec.start_step}:{spec.d}".encode()
            ).hexdigest(),
            spec,
        ),
    )


def select_usage_specs(specs: Mapping[int, Sequence[Layer1Spec]]) -> dict[int, list[Layer1Spec]]:
    selected: dict[int, list[Layer1Spec]] = {}
    for d in (50, 75):
        eligible = [spec for spec in specs[d] if spec.start_step >= CLOCK_STEPS]
        ordered = fixed_hash_order(eligible, "h1-usage-20260808")
        if len(ordered) < 4096:
            raise ValueError("h1_data_blocked: fewer than 4096 usage tasks")
        selected[d] = ordered[:4096]
    return selected
