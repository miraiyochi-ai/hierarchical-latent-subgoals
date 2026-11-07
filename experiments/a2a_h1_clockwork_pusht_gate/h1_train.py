#!/usr/bin/env python3
"""Joint H1 training runner for the registered Mac preflight and GPU runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import shutil
import sys
import tempfile
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from h1_data import (
    ACTION_MAXIMUM,
    ACTION_MINIMUM,
    CyclicSpecSampler,
    EpisodeTable,
    H1HDF5Reader,
    Normalizers,
    collate,
    enumerate_layer1_specs,
    enumerate_layer2_specs,
    layer1_batch_counts,
    layer2_batch_counts,
    load_split_ids,
)
from h1_model import (
    H1ModelConfig,
    H1Policy,
    MODEL_SEEDS,
    build_capacity_matched_pair,
    parameter_manifest,
    save_checkpoint,
)


FULL_UPDATES = 30_000
FULL_BATCH = 128
MAC_UPDATES = 500
MAC_BATCH = 16
DATA_PREFETCH_DEPTH = 4
WORKER_SPEC_CHUNK = 4
LEARNING_RATE = 1e-4
WARMUP_UPDATES = 500
LAYER2_WEIGHT = 1.0


@dataclass(frozen=True)
class RunConfig:
    profile: str
    updates: int
    batch_size: int
    image_size: int
    latent_dim: int
    hidden_dim: int
    seeds: tuple[int, ...]
    device: str


def effective_run_config(profile: str, requested_device: str | None = None) -> RunConfig:
    if profile == "gpu_full":
        device = requested_device or "cuda"
        if device != "cuda":
            raise ValueError("the full registered profile requires CUDA")
        return RunConfig(profile, FULL_UPDATES, FULL_BATCH, 224, 512, 512, MODEL_SEEDS, device)
    if profile == "mac_preflight":
        if requested_device:
            device = requested_device
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if device not in {"mps", "cpu"}:
            raise ValueError("Mac preflight must use MPS or CPU")
        return RunConfig(profile, MAC_UPDATES, MAC_BATCH, 64, 128, 128, (42,), device)
    raise ValueError("profile must be gpu_full or mac_preflight")


def effective_data_workers(profile: str, requested: int | None) -> int:
    if requested is not None:
        if requested < 0:
            raise ValueError("data workers cannot be negative")
        return requested
    if profile == "mac_preflight":
        return 0
    if profile != "gpu_full":
        raise ValueError("profile must be gpu_full or mac_preflight")
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1
    return min(32, cpu_count)


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_multiplier(update: int, total: int) -> float:
    if not 0 <= update < total:
        raise ValueError("update is outside the schedule")
    if update < WARMUP_UPDATES:
        return float(update + 1) / WARMUP_UPDATES
    progress = (update + 1 - WARMUP_UPDATES) / (total - WARMUP_UPDATES)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class PowerEMA:
    """Official power-function EMA: inv_gamma=1, power=.75, [0,.9999]."""

    def __init__(self, model: H1Policy) -> None:
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.optimization_step = 0

    def decay(self) -> float:
        value = 1.0 - (1.0 + self.optimization_step / 1.0) ** -0.75
        return min(0.9999, max(0.0, value))

    @torch.no_grad()
    def step(self, model: H1Policy) -> float:
        decay = self.decay()
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                self.state[name].copy_(value)
        self.optimization_step += 1
        return decay


def move(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    result = {}
    for path in sorted(here.glob("h1_*.py")):
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _take_specs(sampler: CyclicSpecSampler, counts: Mapping[Any, int]) -> list[Any]:
    specs = []
    for key, count in counts.items():
        specs.extend(sampler.take(key, count))
    return specs


def _build_batch_from_specs(reader: H1HDF5Reader, specs: Iterable[Any], layer: int):
    records = [reader.layer1(spec) if layer == 1 else reader.layer2(spec) for spec in specs]
    return collate(records)


class SequentialBatchBuilder:
    """Reference path: select and materialize each registered batch synchronously."""

    def __init__(
        self,
        reader: H1HDF5Reader,
        layer1_sampler: CyclicSpecSampler,
        layer2_sampler: CyclicSpecSampler | None,
        batch_size: int,
    ) -> None:
        self.reader = reader
        self.layer1_sampler = layer1_sampler
        self.layer2_sampler = layer2_sampler
        self.batch_size = batch_size

    def get(self, update: int) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        layer1_specs = _take_specs(
            self.layer1_sampler, layer1_batch_counts(update, self.batch_size)
        )
        layer1 = _build_batch_from_specs(self.reader, layer1_specs, 1)
        layer2 = None
        if self.layer2_sampler is not None:
            layer2_specs = _take_specs(
                self.layer2_sampler, layer2_batch_counts(self.batch_size)
            )
            layer2 = _build_batch_from_specs(self.reader, layer2_specs, 2)
        return layer1, layer2

    def close(self) -> None:
        self.reader.close()


@dataclass(frozen=True)
class _MappedTensorSpec:
    path: str
    shape: tuple[int, ...]
    dtype: str


def _batch_schema(
    layer: int, batch_size: int, image_size: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    image = (batch_size, 8, 3, image_size, image_size)
    if layer == 1:
        return {
            "head_cam": (image, torch.float32),
            "goal_cam": (image, torch.float32),
            "agent_pos": ((batch_size, 8, 2), torch.float32),
            "future_actions": ((batch_size, 8, 2), torch.float32),
            "slow_history_actions": ((batch_size, 4, 8, 2), torch.float32),
            "slow_state_mask": ((batch_size, 1), torch.float32),
            "d": ((batch_size,), torch.int64),
            "episode_idx": ((batch_size,), torch.int64),
            "start_step": ((batch_size,), torch.int64),
        }
    if layer == 2:
        return {
            "history_actions": ((batch_size, 4, 8, 2), torch.float32),
            "target_actions": ((batch_size, 8, 2), torch.float32),
            "goal_cam": (image, torch.float32),
            "d": ((batch_size,), torch.int64),
            "boundary": ((batch_size,), torch.int64),
        }
    raise ValueError("batch layer must be one or two")


def _map_tensor(spec: _MappedTensorSpec) -> Tensor:
    dtype = {"float32": torch.float32, "int64": torch.int64}[spec.dtype]
    return torch.from_file(
        spec.path,
        shared=True,
        size=math.prod(spec.shape),
        dtype=dtype,
    ).reshape(spec.shape)


def _create_mapped_batch(
    root: Path,
    *,
    layer: int,
    slot: int,
    batch_size: int,
    image_size: int,
) -> tuple[dict[str, Tensor], dict[str, _MappedTensorSpec]]:
    tensors: dict[str, Tensor] = {}
    specs: dict[str, _MappedTensorSpec] = {}
    for name, (shape, dtype) in _batch_schema(layer, batch_size, image_size).items():
        path = root / f"layer{layer}_slot{slot}_{name}.bin"
        path.touch(mode=0o600)
        dtype_name = "float32" if dtype == torch.float32 else "int64"
        spec = _MappedTensorSpec(str(path), shape, dtype_name)
        tensors[name] = _map_tensor(spec)
        specs[name] = spec
    return tensors, specs


def _open_mapped_batch(specs: Mapping[str, _MappedTensorSpec]) -> dict[str, Tensor]:
    return {name: _map_tensor(spec) for name, spec in specs.items()}


def _mapped_ring_bytes(
    batch_size: int, image_size: int, hierarchical: bool, prefetch_depth: int
) -> int:
    layers = (1, 2) if hierarchical else (1,)
    per_slot = sum(
        math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        for layer in layers
        for shape, dtype in _batch_schema(layer, batch_size, image_size).values()
    )
    return prefetch_depth * per_slot


def _new_mapped_root(required_bytes: int) -> Path:
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
        free = shutil.disk_usage(shared_memory).free
        if free >= required_bytes + 64 * 1024 * 1024:
            return Path(tempfile.mkdtemp(prefix="h1-batch-ring-", dir=shared_memory))
    return Path(tempfile.mkdtemp(prefix="h1-batch-ring-"))


def _batch_worker_main(
    connection: Connection,
    hdf5: str,
    lengths: np.ndarray,
    offsets: np.ndarray,
    normalizers: Normalizers,
    image_size: int,
    layer1_specs: tuple[dict[str, _MappedTensorSpec], ...],
    layer2_specs: tuple[dict[str, _MappedTensorSpec], ...] | None,
) -> None:
    """Write records directly into mapped batch slots and send only completion IDs."""
    torch.set_num_threads(1)
    table = EpisodeTable(np.asarray(lengths), np.asarray(offsets))
    reader = H1HDF5Reader(
        Path(hdf5), table, normalizers, image_size=image_size
    )
    layer1_slots = tuple(_open_mapped_batch(specs) for specs in layer1_specs)
    layer2_slots = (
        tuple(_open_mapped_batch(specs) for specs in layer2_specs)
        if layer2_specs is not None
        else None
    )
    try:
        reader.open()
        while True:
            message = connection.recv()
            if message is None:
                break
            task_id, layer, slot, row_start, specs = message
            try:
                if layer == 1:
                    destination = layer1_slots[slot]
                elif layer2_slots is None:
                    raise RuntimeError("flat worker received a layer-two task")
                else:
                    destination = layer2_slots[slot]
                for row_offset, spec in enumerate(specs):
                    record = reader.layer1(spec) if layer == 1 else reader.layer2(spec)
                    row = row_start + row_offset
                    for name, value in record.items():
                        destination[name][row].copy_(value)
                connection.send((task_id, None))
            except BaseException:
                connection.send((task_id, traceback.format_exc()))
    finally:
        reader.close()
        connection.close()


@dataclass(frozen=True)
class _WorkerSlot:
    connection: Connection
    process: Any


@dataclass(frozen=True)
class _PendingTask:
    task_id: int
    worker_index: int


@dataclass(frozen=True)
class _PendingUpdate:
    update: int
    slot: int
    layer1: tuple[_PendingTask, ...]
    layer2: tuple[_PendingTask, ...] | None


class OrderedParallelBatchBuilder:
    """Parallel materialization with main-process sampling and ordered consumption."""

    def __init__(
        self,
        *,
        hdf5: Path,
        table: EpisodeTable,
        normalizers: Normalizers,
        image_size: int,
        layer1_sampler: CyclicSpecSampler,
        layer2_sampler: CyclicSpecSampler | None,
        batch_size: int,
        total_updates: int,
        workers: int,
        prefetch_depth: int = DATA_PREFETCH_DEPTH,
    ) -> None:
        if workers <= 0 or prefetch_depth <= 0 or total_updates <= 0:
            raise ValueError("parallel batch builder requires positive bounds")
        self.layer1_sampler = layer1_sampler
        self.layer2_sampler = layer2_sampler
        self.batch_size = batch_size
        self.total_updates = total_updates
        self.prefetch_depth = prefetch_depth
        self.next_submit = 0
        self.next_consume = 0
        self.next_task_id = 0
        self.next_worker = 0
        self.pending: deque[_PendingUpdate] = deque()
        self.active_slot: int | None = None
        self.shared_root = _new_mapped_root(
            _mapped_ring_bytes(
                batch_size, image_size, layer2_sampler is not None, prefetch_depth
            )
        )
        try:
            layer1_pairs = tuple(
                _create_mapped_batch(
                    self.shared_root,
                    layer=1,
                    slot=slot,
                    batch_size=batch_size,
                    image_size=image_size,
                )
                for slot in range(prefetch_depth)
            )
            self.layer1_slots = tuple(pair[0] for pair in layer1_pairs)
            layer1_specs = tuple(pair[1] for pair in layer1_pairs)
            if layer2_sampler is not None:
                layer2_pairs = tuple(
                    _create_mapped_batch(
                        self.shared_root,
                        layer=2,
                        slot=slot,
                        batch_size=batch_size,
                        image_size=image_size,
                    )
                    for slot in range(prefetch_depth)
                )
                self.layer2_slots: tuple[dict[str, Tensor], ...] | None = tuple(
                    pair[0] for pair in layer2_pairs
                )
                layer2_specs: tuple[dict[str, _MappedTensorSpec], ...] | None = tuple(
                    pair[1] for pair in layer2_pairs
                )
            else:
                self.layer2_slots = None
                layer2_specs = None
        except BaseException:
            shutil.rmtree(self.shared_root, ignore_errors=True)
            raise
        context = multiprocessing.get_context("spawn")
        self.workers: list[_WorkerSlot] = []
        try:
            for worker_index in range(workers):
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    name=f"h1-data-{worker_index:02d}",
                    target=_batch_worker_main,
                    args=(
                        child_connection,
                        str(Path(hdf5).resolve()),
                        table.lengths,
                        table.offsets,
                        normalizers,
                        image_size,
                        layer1_specs,
                        layer2_specs,
                    ),
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent_connection.close()
                    child_connection.close()
                    raise
                child_connection.close()
                self.workers.append(_WorkerSlot(parent_connection, process))
            self._fill()
        except BaseException:
            self.close()
            raise

    def _submit_layer(
        self, layer: int, slot: int, specs: list[Any]
    ) -> tuple[_PendingTask, ...]:
        tasks = []
        for start in range(0, len(specs), WORKER_SPEC_CHUNK):
            worker_index = self.next_worker
            task = _PendingTask(self.next_task_id, worker_index)
            self.workers[worker_index].connection.send(
                (
                    task.task_id,
                    layer,
                    slot,
                    start,
                    tuple(specs[start : start + WORKER_SPEC_CHUNK]),
                )
            )
            tasks.append(task)
            self.next_task_id += 1
            self.next_worker = (self.next_worker + 1) % len(self.workers)
        return tuple(tasks)

    def _submit(self, update: int) -> None:
        slot = update % self.prefetch_depth
        if self.active_slot == slot or any(
            pending.slot == slot for pending in self.pending
        ):
            raise RuntimeError("mapped batch slot was reused before consumption")
        layer1_specs = _take_specs(
            self.layer1_sampler, layer1_batch_counts(update, self.batch_size)
        )
        layer1 = self._submit_layer(1, slot, layer1_specs)
        layer2 = None
        if self.layer2_sampler is not None:
            layer2_specs = _take_specs(
                self.layer2_sampler, layer2_batch_counts(self.batch_size)
            )
            layer2 = self._submit_layer(2, slot, layer2_specs)
        self.pending.append(_PendingUpdate(update, slot, layer1, layer2))

    def _fill(self) -> None:
        while len(self.pending) < self.prefetch_depth and self.next_submit < self.total_updates:
            self._submit(self.next_submit)
            self.next_submit += 1

    def _resolve(self, tasks: tuple[_PendingTask, ...]) -> None:
        for task in tasks:
            task_id, error = self.workers[task.worker_index].connection.recv()
            if task_id != task.task_id:
                raise RuntimeError("data worker result order changed")
            if error is not None:
                raise RuntimeError(f"data worker failed:\n{error}")

    def get(self, update: int) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        if self.active_slot is not None:
            self.active_slot = None
            self._fill()
        if update != self.next_consume or not self.pending:
            raise RuntimeError("parallel batches must be consumed in registered update order")
        pending = self.pending.popleft()
        if pending.update != update:
            raise RuntimeError("prefetched batch update order changed")
        self._resolve(pending.layer1)
        if pending.layer2 is not None:
            self._resolve(pending.layer2)
        layer1 = self.layer1_slots[pending.slot]
        layer2 = (
            self.layer2_slots[pending.slot]
            if self.layer2_slots is not None
            else None
        )
        self.active_slot = pending.slot
        self.next_consume += 1
        return layer1, layer2

    def close(self) -> None:
        incomplete = self.next_consume < self.total_updates
        for worker in self.workers:
            if incomplete:
                if worker.process.is_alive():
                    worker.process.terminate()
            elif worker.process.is_alive():
                worker.connection.send(None)
        for worker in self.workers:
            worker.process.join()
        for worker in self.workers:
            worker.connection.close()
        self.workers.clear()
        self.layer1_slots = ()
        self.layer2_slots = None
        shutil.rmtree(self.shared_root, ignore_errors=True)


def _microbatches(batch: Mapping[str, Tensor], size: int) -> Iterable[dict[str, Tensor]]:
    count = int(next(iter(batch.values())).shape[0])
    if size <= 0 or size > count or count % size:
        raise ValueError("microbatch size must divide the registered effective batch")
    for start in range(0, count, size):
        yield {name: value[start : start + size] for name, value in batch.items()}


def train_one(
    model: H1Policy,
    *,
    run_config: RunConfig,
    batch_builder: SequentialBatchBuilder | OrderedParallelBatchBuilder,
    loss_path: Path,
    microbatch_size: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    device = torch.device(run_config.device)
    model.to(device).train()
    generator = model.new_generator(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )
    ema = PowerEMA(model)
    layer1_nonzero = False
    layer2_nonzero = not model.config.hierarchical
    nonfinite = 0
    for update in range(run_config.updates):
        multiplier = learning_rate_multiplier(update, run_config.updates)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * multiplier
        optimizer.zero_grad(set_to_none=True)
        l1_batch, l2_batch = batch_builder.get(update)
        component_values = {name: 0.0 for name in ("total", "flow", "kl")}
        for microbatch in _microbatches(l1_batch, microbatch_size):
            moved = move(microbatch, device)
            components = model.layer1_loss(moved, generator=generator)
            if not torch.isfinite(components["total"]):
                nonfinite += 1
                raise RuntimeError("non-finite H1 layer-one loss")
            weight = int(moved["future_actions"].shape[0]) / run_config.batch_size
            (components["total"] * weight).backward()
            for name in component_values:
                component_values[name] += float(components[name].detach().cpu()) * weight
        if model.config.hierarchical:
            if l2_batch is None:
                raise RuntimeError("hierarchical training requires a layer-two batch")
            layer2_value = 0.0
            for microbatch in _microbatches(l2_batch, microbatch_size):
                moved = move(microbatch, device)
                layer2_loss = model.layer2_loss(moved, generator=generator)
                if not torch.isfinite(layer2_loss):
                    nonfinite += 1
                    raise RuntimeError("non-finite H1 layer-two loss")
                weight = int(moved["target_actions"].shape[0]) / run_config.batch_size
                (LAYER2_WEIGHT * layer2_loss * weight).backward()
                layer2_value += float(layer2_loss.detach().cpu()) * weight
        else:
            if l2_batch is not None:
                raise RuntimeError("flat training must not materialize layer-two batches")
            layer2_value = 0.0
        total_value = component_values["total"] + LAYER2_WEIGHT * layer2_value
        if not math.isfinite(total_value):
            nonfinite += 1
            raise RuntimeError("non-finite H1 total loss")
        layer1_this_update = any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.layer1_velocity.parameters()
        )
        layer1_nonzero |= layer1_this_update
        layer2_this_update = False
        if model.layer2 is not None:
            layer2_this_update = any(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                for parameter in model.layer2.parameters()
            )
            layer2_nonzero |= layer2_this_update
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        decay = ema.step(model)
        _append_jsonl(
            loss_path,
            {
                "profile": run_config.profile,
                "seed": model.config.seed,
                "condition": "hierarchical" if model.config.hierarchical else "flat",
                "update": update + 1,
                "total": total_value,
                "layer1": component_values["total"],
                "layer2": layer2_value,
                "flow": component_values["flow"],
                "kl": component_values["kl"],
                "grad_norm": float(torch.as_tensor(grad_norm).cpu()),
                "layer1_gradient_nonzero": layer1_this_update,
                "layer2_gradient_nonzero": layer2_this_update,
                "lr": optimizer.param_groups[0]["lr"],
                "ema_decay": decay,
            },
        )
    return ema.state, {
        "updates": run_config.updates,
        "nonfinite_loss_count": nonfinite,
        "layer1_gradient_seen": layer1_nonzero,
        "layer2_gradient_seen": layer2_nonzero,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("mac_preflight", "gpu_full"), required=True)
    parser.add_argument("--condition", choices=("all", "hierarchical", "flat"), default="all")
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--data-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_config = effective_run_config(args.profile, args.device)
    if args.seed is not None:
        if args.profile != "gpu_full":
            raise ValueError("--seed is only a distributed full-GPU execution option")
        run_config = replace(run_config, seeds=(args.seed,))
    microbatch_size = args.microbatch_size or run_config.batch_size
    if microbatch_size <= 0 or run_config.batch_size % microbatch_size:
        raise ValueError("microbatch size must be a positive divisor of the effective batch")
    data_workers = effective_data_workers(args.profile, args.data_workers)
    stage0_verification = args.data_root.parent / "verification.json"
    if not stage0_verification.is_file():
        raise RuntimeError("h1_data_blocked: independent stage-zero verification is missing")
    stage0 = json.loads(stage0_verification.read_text(encoding="utf-8"))
    if stage0.get("verdict") != "h1_stage0_pass" or not all(stage0.get("gates", {}).values()):
        raise RuntimeError("h1_data_blocked: stage zero did not pass every fixed gate")
    action_contract = stage0.get("action_contract", {})
    minimum = np.asarray(action_contract.get("minimum_by_dimension", []), dtype=np.float64)
    maximum = np.asarray(action_contract.get("maximum_by_dimension", []), dtype=np.float64)
    if (
        minimum.shape != (2,)
        or maximum.shape != (2,)
        or not np.array_equal(minimum.astype(np.float32), ACTION_MINIMUM)
        or not np.array_equal(maximum.astype(np.float32), ACTION_MAXIMUM)
    ):
        raise RuntimeError("h1_data_blocked: audited action extrema differ from the corrected fixed values")
    splits = load_split_ids(args.data_root / "split_ids.json")
    table = EpisodeTable.from_hdf5(args.hdf5)
    normalizers = Normalizers.load(args.data_root / "normalizers.npz")
    train_ids = splits["train"]
    if args.profile == "mac_preflight":
        train_ids = train_ids[:64]
    l1_pools = enumerate_layer1_specs(table, train_ids)
    l2_pools = enumerate_layer2_specs(table, train_ids)
    output = args.output_root
    loss_path = output / (
        "mac_preflight/raw_losses.jsonl" if args.profile == "mac_preflight" else "training_losses.jsonl"
    )
    if loss_path.exists():
        raise FileExistsError(f"refusing to append to an existing formal loss log: {loss_path}")
    environment_name = (
        "environment_manifest_gpu_train.json" if args.profile == "gpu_full" else "environment_manifest_mac.json"
    )
    environment_path = output / environment_name
    if not environment_path.exists():
        _atomic_json(
            environment_path,
            {
                "profile": args.profile,
                "platform": platform.platform(),
                "python": sys.version,
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "device": run_config.device,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if run_config.device == "cuda" else None,
                "implementation_sha256": source_hashes(),
                "microbatch_size": microbatch_size,
                "effective_optimizer_batch_size": run_config.batch_size,
                "data_workers": data_workers,
                "data_prefetch_depth": DATA_PREFETCH_DEPTH if data_workers else 0,
            },
        )
    conditions = ("hierarchical", "flat") if args.condition == "all" else (args.condition,)
    for seed in run_config.seeds:
        configure_seed(seed)
        base_config = (
            H1ModelConfig.full(seed=seed, hierarchical=True)
            if args.profile == "gpu_full"
            else H1ModelConfig.mac_preflight(seed=seed, hierarchical=True)
        )
        hierarchical, flat, ledger = build_capacity_matched_pair(base_config)
        manifest_path = output / "parameter_manifest.json"
        if args.profile == "gpu_full" and not manifest_path.exists():
            _atomic_json(manifest_path, ledger)
        if args.profile == "mac_preflight":
            mac_config = output / "mac_preflight" / "effective_config.json"
            if not mac_config.exists():
                _atomic_json(
                    mac_config,
                    {
                        "run": asdict(run_config),
                        "capacity": ledger,
                        "optimizer": {
                            "name": "AdamW",
                            "lr": LEARNING_RATE,
                            "betas": [0.95, 0.999],
                            "eps": 1e-8,
                            "weight_decay": 1e-6,
                        },
                        "warmup_updates": WARMUP_UPDATES,
                        "schedule": "cosine",
                        "gradient_norm_limit": 1.0,
                        "layer2_weight": LAYER2_WEIGHT,
                        "training_episode_ids": train_ids.tolist(),
                        "validation_episode_ids": splits["validation"][:16].tolist(),
                        "split_ids_sha256": hashlib.sha256(
                            (args.data_root / "split_ids.json").read_bytes()
                        ).hexdigest(),
                        "normalizers_sha256": hashlib.sha256(
                            (args.data_root / "normalizers.npz").read_bytes()
                        ).hexdigest(),
                        "implementation_sha256": source_hashes(),
                        "data_workers": data_workers,
                        "data_prefetch_depth": DATA_PREFETCH_DEPTH if data_workers else 0,
                    },
                )
        for condition in conditions:
            model = hierarchical if condition == "hierarchical" else flat
            effective = {
                **asdict(run_config),
                "model": asdict(model.config),
                "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "betas": [0.95, 0.999], "eps": 1e-8, "weight_decay": 1e-6},
                "warmup_updates": WARMUP_UPDATES,
                "schedule": "cosine",
                "gradient_norm_limit": 1.0,
                "layer2_weight": LAYER2_WEIGHT,
                "implementation_sha256": source_hashes(),
                "microbatch_size": microbatch_size,
                "effective_optimizer_batch_size": run_config.batch_size,
                "data_workers": data_workers,
                "data_prefetch_depth": DATA_PREFETCH_DEPTH if data_workers else 0,
            }
            if args.profile == "gpu_full":
                _atomic_json(output / "effective_configs" / f"{condition}_seed_{seed}.json", effective)
            l1_sampler = CyclicSpecSampler(l1_pools, seed)
            l2_sampler = CyclicSpecSampler(l2_pools, seed) if condition == "hierarchical" else None
            if data_workers:
                batch_builder: SequentialBatchBuilder | OrderedParallelBatchBuilder = (
                    OrderedParallelBatchBuilder(
                        hdf5=args.hdf5,
                        table=table,
                        normalizers=normalizers,
                        image_size=run_config.image_size,
                        layer1_sampler=l1_sampler,
                        layer2_sampler=l2_sampler,
                        batch_size=run_config.batch_size,
                        total_updates=run_config.updates,
                        workers=data_workers,
                    )
                )
            else:
                batch_builder = SequentialBatchBuilder(
                    H1HDF5Reader(
                        args.hdf5,
                        table,
                        normalizers,
                        image_size=run_config.image_size,
                    ),
                    l1_sampler,
                    l2_sampler,
                    run_config.batch_size,
                )
            try:
                ema_state, summary = train_one(
                    model,
                    run_config=run_config,
                    batch_builder=batch_builder,
                    loss_path=loss_path,
                    microbatch_size=microbatch_size,
                )
            finally:
                batch_builder.close()
            if args.profile == "gpu_full":
                checkpoint = output / "checkpoints" / f"seed_{seed}" / f"{condition}_ema.pt"
            else:
                checkpoint = output / "mac_preflight" / "checkpoints" / f"{condition}_ema.pt"
            save_checkpoint(checkpoint, model, update=run_config.updates, ema_state=ema_state)
            _atomic_json(checkpoint.with_suffix(".summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
