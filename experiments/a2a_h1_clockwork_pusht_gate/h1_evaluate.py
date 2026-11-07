#!/usr/bin/env python3
"""Closed-loop H1 evaluation with the five registered policy conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hdf5plugin  # noqa: F401  # Registers the compression filters used by the fixed HDF5.
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from h1_data import EpisodeTable, H1HDF5Reader, Layer1Spec, Normalizers
from h1_model import ACTION_STEPS, CLOCK_CHUNKS, H1Policy, load_checkpoint


CONDITIONS = ("present", "zero", "swap", "flat", "nearest_demo")
BUDGETS = {25: 50, 50: 100, 75: 150}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_environment_once(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("h1_infrastructure_blocked: evaluation environments differ")


def fixed_cyclic_partners(tasks: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    """Return sorted task indices and a next-task cyclic partner for every row."""
    if len(tasks) < 2:
        raise ValueError("swap requires at least two episodes")
    order = sorted(range(len(tasks)), key=lambda i: (int(tasks[i]["episode_idx"]), int(tasks[i]["start_step"])))
    partner = [-1] * len(tasks)
    for position, index in enumerate(order):
        partner[index] = order[(position + 1) % len(order)]
    if any(index == partner[index] for index in range(len(tasks))):
        raise RuntimeError("h1_infrastructure_blocked: cyclic swap contains self correspondence")
    return order, partner


def execution_length(remaining_budget: int) -> int:
    if remaining_budget <= 0:
        return 0
    return min(ACTION_STEPS, remaining_budget)


def completes_clock_update(executed_chunks: int, last_chunk_length: int) -> bool:
    return last_chunk_length == ACTION_STEPS and executed_chunks > 0 and executed_chunks % CLOCK_CHUNKS == 0


@dataclass
class EpisodeClock:
    latent_dim: int
    executed_chunks: list[np.ndarray] = field(default_factory=list)
    slow_state: np.ndarray | None = None
    raw_slow_state: np.ndarray | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        self.slow_state = np.zeros(self.latent_dim, dtype=np.float32)
        self.raw_slow_state = np.zeros(self.latent_dim, dtype=np.float32)

    def add_chunk(self, actions: np.ndarray) -> bool:
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (ACTION_STEPS, 2):
            return False
        self.executed_chunks.append(values.copy())
        return len(self.executed_chunks) % CLOCK_CHUNKS == 0

    def latest_history(self) -> np.ndarray:
        if len(self.executed_chunks) < CLOCK_CHUNKS:
            raise RuntimeError("four full chunks are required")
        return np.stack(self.executed_chunks[-CLOCK_CHUNKS:])


def area_feature(current: np.ndarray, goal: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    current_tensor = torch.as_tensor(current, dtype=torch.float32)
    goal_tensor = torch.as_tensor(goal, dtype=torch.float32)
    if current_tensor.ndim == 3 and current_tensor.shape[-1] == 3:
        current_tensor = current_tensor.permute(2, 0, 1)
    if goal_tensor.ndim == 3 and goal_tensor.shape[-1] == 3:
        goal_tensor = goal_tensor.permute(2, 0, 1)
    if float(current_tensor.max()) > 1.0:
        current_tensor = current_tensor / 255.0
    if float(goal_tensor.max()) > 1.0:
        goal_tensor = goal_tensor / 255.0
    pooled = [F.adaptive_avg_pool2d(image[None], (16, 16)).flatten() for image in (current_tensor, goal_tensor)]
    return torch.cat((*pooled, torch.as_tensor(agent_pos, dtype=torch.float32))).numpy()


class NearestDemonstration:
    def __init__(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        episode_idx: np.ndarray,
        step_idx: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.episode_idx = np.asarray(episode_idx, dtype=np.int64)
        self.step_idx = np.asarray(step_idx, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if np.any(self.std <= 0) or self.actions.shape[1:] != (8, 2):
            raise ValueError("invalid nearest-demonstration bank")
        self.standard = (self.features - self.mean) / self.std

    def query(self, feature: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        standardized = (np.asarray(feature, dtype=np.float32) - self.mean) / self.std
        distance = np.sum((self.standard - standardized) ** 2, axis=1)
        candidates = np.flatnonzero(distance == distance.min())
        index = min(candidates, key=lambda i: (int(self.episode_idx[i]), int(self.step_idx[i])))
        return self.actions[index].copy(), (int(self.episode_idx[index]), int(self.step_idx[index]))


def build_nearest_demo_bank(
    reader: H1HDF5Reader, specs: Sequence[Layer1Spec], output: Path
) -> dict[str, Any]:
    """Materialize one horizon's training-only bank with deterministic tie metadata."""
    if not specs or len({spec.d for spec in specs}) != 1:
        raise ValueError("nearest-demonstration bank must contain one nonempty horizon")
    if output.exists():
        raise FileExistsError("refusing to overwrite nearest-demonstration bank")
    features, actions, episodes, steps = [], [], [], []
    for spec in sorted(specs):
        record = reader.layer1(spec)
        current = record["head_cam"][-1].numpy()
        goal = record["goal_cam"][0].numpy()
        agent = record["agent_pos"][-1].numpy()
        features.append(area_feature(current, goal, agent))
        actions.append(record["future_actions"].numpy())
        episodes.append(spec.episode_idx)
        steps.append(spec.start_step)
    feature_array = np.asarray(features, dtype=np.float32)
    mean = feature_array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = feature_array.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        features=feature_array,
        actions=np.asarray(actions, dtype=np.float32),
        episode_idx=np.asarray(episodes, dtype=np.int64),
        step_idx=np.asarray(steps, dtype=np.int64),
        mean=mean,
        std=std,
    )
    return {"d": specs[0].d, "count": len(specs), "sha256": sha256_file(output)}


class ClockedBatchPolicy:
    """Synchronous 50-episode planner; only actually executed chunks enter layer two."""

    def __init__(
        self,
        model: H1Policy,
        condition: str,
        goals: np.ndarray,
        initial_images: np.ndarray,
        initial_agent: np.ndarray,
        normalizers: Normalizers,
        tasks: Sequence[Mapping[str, Any]],
        device: torch.device,
        nearest: NearestDemonstration | None = None,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError("unregistered evaluation condition")
        if condition == "flat" and model.config.hierarchical:
            raise ValueError("flat condition requires the flat checkpoint")
        if condition != "flat" and condition != "nearest_demo" and not model.config.hierarchical:
            raise ValueError("hierarchical condition requires the hierarchical checkpoint")
        self.model = model.to(device).eval()
        self.condition = condition
        self.goals = goals
        self.normalizers = normalizers
        self.device = device
        self.nearest = nearest
        self.tasks = tasks
        self.order, self.partners = fixed_cyclic_partners(tasks)
        self.image_history = [[image.copy()] * 8 for image in initial_images]
        self.agent_history = [[agent.copy()] * 8 for agent in initial_agent]
        self.clocks = [EpisodeClock(model.config.latent_dim) for _ in tasks]
        self.queues: list[list[np.ndarray]] = [[] for _ in tasks]
        self.pending_chunks: list[list[np.ndarray]] = [[] for _ in tasks]
        self.slow_events: list[list[dict[str, Any]]] = [[] for _ in tasks]
        self.nearest_refs: list[list[tuple[int, int]]] = [[] for _ in tasks]

    def _model_images(self, values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(values).to(self.device, dtype=torch.float32)
        batch, steps = tensor.shape[:2]
        tensor = tensor.reshape(batch * steps, *tensor.shape[2:]).div(255.0)
        size = self.model.config.image_size
        if tensor.shape[-2:] != (size, size):
            tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        return tensor.reshape(batch, steps, 3, size, size)

    def observe(self, images: np.ndarray, agents: np.ndarray, active: np.ndarray) -> None:
        for i in range(len(self.tasks)):
            if active[i]:
                self.image_history[i].append(images[i].copy())
                self.image_history[i] = self.image_history[i][-8:]
                self.agent_history[i].append(agents[i].copy())
                self.agent_history[i] = self.agent_history[i][-8:]

    def _update_slow_states(self, indices: Sequence[int]) -> None:
        if not indices or self.condition in {"flat", "nearest_demo"}:
            return
        histories = torch.from_numpy(np.stack([self.clocks[i].latest_history() for i in indices])).to(self.device)
        goals = np.stack([np.repeat(self.goals[i][None], 8, axis=0) for i in indices])
        goal_tensor = self._model_images(goals)
        with torch.no_grad():
            goal_condition = self.model.goal_encoder(goal_tensor)
            raw = self.model.predict_slow_state(histories, goal_condition).cpu().numpy()
        for local, index in enumerate(indices):
            previous = self.clocks[index].raw_slow_state.copy()
            self.clocks[index].raw_slow_state = raw[local]
            self.clocks[index].updates += 1
            self.slow_events[index].append(
                {
                    "clock_update": self.clocks[index].updates,
                    "norm": float(np.linalg.norm(raw[local])),
                    "update_norm": float(np.linalg.norm(raw[local] - previous)),
                }
            )
        if self.condition == "present":
            for index in indices:
                self.clocks[index].slow_state = self.clocks[index].raw_slow_state.copy()
        elif self.condition == "zero":
            for index in indices:
                self.clocks[index].slow_state.fill(0)
        elif self.condition == "swap":
            raw_all = [clock.raw_slow_state.copy() for clock in self.clocks]
            for index in indices:
                self.clocks[index].slow_state = raw_all[self.partners[index]].copy()

    def commit_completed(self, active: np.ndarray) -> None:
        update_indices = []
        for i in range(len(self.tasks)):
            if len(self.pending_chunks[i]) == ACTION_STEPS:
                if self.clocks[i].add_chunk(np.stack(self.pending_chunks[i])):
                    update_indices.append(i)
                self.pending_chunks[i].clear()
        self._update_slow_states(update_indices)

    def plan(self, active: np.ndarray) -> None:
        indices = [i for i in range(len(self.tasks)) if active[i] and not self.queues[i]]
        if not indices:
            return
        if self.condition == "nearest_demo":
            if self.nearest is None:
                raise RuntimeError("nearest-demo condition has no training bank")
            for i in indices:
                feature = area_feature(self.image_history[i][-1], self.goals[i], self.agent_history[i][-1])
                actions, ref = self.nearest.query(feature)
                self.queues[i] = list(actions)
                self.nearest_refs[i].append(ref)
            return
        head = np.stack([np.stack(self.image_history[i]) for i in indices])
        goal = np.stack([np.repeat(self.goals[i][None], 8, axis=0) for i in indices])
        agent = np.stack([self.normalizers.agent(np.stack(self.agent_history[i])) for i in indices])
        slow = None
        if self.model.config.hierarchical:
            slow = torch.from_numpy(np.stack([self.clocks[i].slow_state for i in indices])).to(self.device)
        with torch.no_grad():
            actions = self.model.predict_actions(
                self._model_images(head),
                self._model_images(goal),
                torch.from_numpy(agent).to(self.device),
                slow,
            ).cpu().numpy()
        if not np.isfinite(actions).all():
            raise RuntimeError("h1_infrastructure_blocked: policy emitted a non-finite action")
        for local, index in enumerate(indices):
            self.queues[index] = list(actions[local])

    def next_actions(self, active: np.ndarray) -> np.ndarray:
        self.commit_completed(active)
        self.plan(active)
        output = np.zeros((len(self.tasks), 2), dtype=np.float32)
        for i in range(len(self.tasks)):
            if active[i]:
                action = np.asarray(self.queues[i].pop(0), dtype=np.float32)
                output[i] = action
                self.pending_chunks[i].append(action.copy())
        return output


def load_tasks(path: Path, d: int, seed: int, *, count: int = 50) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    tasks = [task for task in tasks if int(task["d"]) == d and int(task["seed"]) == seed]
    if len(tasks) != 50:
        raise ValueError("evaluation group must contain exactly 50 tasks")
    return sorted(tasks, key=lambda task: (int(task["episode_idx"]), int(task["start_step"])))[:count]


def checkpoint_set_manifest(checkpoint: Path, *, mac_preflight: bool) -> dict[str, str]:
    checkpoint = Path(checkpoint).resolve()
    if mac_preflight:
        root = checkpoint.parent
        expected = [root / f"{condition}_ema.pt" for condition in ("hierarchical", "flat")]
    else:
        root = checkpoint.parents[1]
        expected = [
            root / f"seed_{seed}" / f"{condition}_ema.pt"
            for seed in (42, 43, 44)
            for condition in ("hierarchical", "flat")
        ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"h1_infrastructure_blocked: final checkpoint set is incomplete: {missing}")
    return {str(path): sha256_file(path) for path in expected}


def _states_and_goals(hdf5: Path, tasks: Sequence[Mapping[str, Any]], table: EpisodeTable):
    starts, goals = [], []
    with h5py.File(hdf5, "r", swmr=True) as handle:
        for task in tasks:
            start = table.global_row(int(task["episode_idx"]), int(task["start_step"]))
            goal = table.global_row(int(task["episode_idx"]), int(task["goal_step"]))
            starts.append(np.asarray(handle["state"][start], dtype=np.float64))
            goals.append(np.asarray(handle["state"][goal], dtype=np.float64))
    return np.stack(starts), np.stack(goals)


def run_group(
    *,
    model: H1Policy,
    condition: str,
    tasks: Sequence[Mapping[str, Any]],
    hdf5: Path,
    normalizers: Normalizers,
    checkpoint_sha256: str,
    input_sha256: str,
    device: torch.device,
    nearest: NearestDemonstration | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from stable_worldmodel.envs.pusht.env import PushT

    table = EpisodeTable.from_hdf5(hdf5)
    start_states, goal_states = _states_and_goals(hdf5, tasks, table)
    envs, images, agents, goal_images = [], [], [], []
    for start, goal in zip(start_states, goal_states):
        env = PushT(resolution=224, relative=True, render_mode="rgb_array")
        env.reset(seed=None)
        env._set_state(goal.copy())
        env._set_goal_state(goal.copy())
        goal_images.append(np.asarray(env.render()).transpose(2, 0, 1))
        env._set_state(start.copy())
        envs.append(env)
        images.append(np.asarray(env.render()).transpose(2, 0, 1))
        agents.append(np.asarray(env._get_obs()[:2], dtype=np.float32))
    goal_images = np.stack(goal_images)
    policy = ClockedBatchPolicy(
        model, condition, goal_images, np.stack(images), np.stack(agents), normalizers, tasks, device, nearest
    )
    n = len(tasks)
    active = np.ones(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    executed = np.zeros(n, dtype=np.int64)
    first_success = np.full(n, -1, dtype=np.int64)
    final_position = np.full(n, np.nan)
    final_angle = np.full(n, np.nan)
    actions = np.zeros((n, int(tasks[0]["eval_budget"]), 2), dtype=np.float32)
    started = [datetime.now(timezone.utc).isoformat() for _ in tasks]
    for step in range(int(tasks[0]["eval_budget"])):
        batch_actions = policy.next_actions(active)
        new_images = np.stack(images)
        new_agents = np.stack(agents)
        for i, env in enumerate(envs):
            if not active[i]:
                continue
            actions[i, step] = batch_actions[i]
            _, _, terminated, truncated, info = env.step(batch_actions[i])
            executed[i] += 1
            images[i] = np.asarray(env.render()).transpose(2, 0, 1)
            agents[i] = np.asarray(env._get_obs()[:2], dtype=np.float32)
            new_images[i] = images[i]
            new_agents[i] = agents[i]
            if terminated:
                success[i] = True
                first_success[i] = step + 1
            active[i] = not (terminated or truncated)
            state = env._get_obs()
            final_position[i] = float(np.linalg.norm(goal_states[i, :4] - state[:4]))
            angle_delta = (float(state[4]) - float(goal_states[i, 4]) + math.pi) % (2 * math.pi) - math.pi
            final_angle[i] = abs(angle_delta)
        policy.observe(new_images, new_agents, active)
        if not active.any():
            break
    ended = datetime.now(timezone.utc).isoformat()
    records = []
    action_arrays: dict[str, np.ndarray] = {}
    for i, task in enumerate(tasks):
        envs[i].close()
        executed_actions = actions[i, : executed[i]]
        action_sha = hashlib.sha256(np.ascontiguousarray(executed_actions).view(np.uint8)).hexdigest()
        action_key = (
            f"{condition}_d{int(task['d'])}_m{model.config.seed}_e{int(task['seed'])}_"
            f"episode{int(task['episode_idx'])}_start{int(task['start_step'])}"
        )
        action_arrays[action_key] = executed_actions
        records.append(
            {
                "record_type": "h1_closed_loop_episode",
                "condition": condition,
                "d": int(task["d"]),
                "model_seed": model.config.seed,
                "evaluation_seed": int(task["seed"]),
                "episode_idx": int(task["episode_idx"]),
                "start_step": int(task["start_step"]),
                "goal_step": int(task["goal_step"]),
                "success": bool(success[i]),
                "termination_reason": "environment_terminated" if success[i] else "budget_exhausted",
                "executed_steps": int(executed[i]),
                "success_step": None if first_success[i] < 0 else int(first_success[i]),
                "clock_updates": policy.clocks[i].updates,
                "swap_partner": None if condition != "swap" else {
                    "episode_idx": int(tasks[policy.partners[i]]["episode_idx"]),
                    "start_step": int(tasks[policy.partners[i]]["start_step"]),
                },
                "slow_state_reference": (
                    "none_flat" if condition == "flat" else
                    "zero_vector" if condition == "zero" else
                    "cyclic_partner" if condition == "swap" else
                    "self_prediction"
                ),
                "slow_state_events": policy.slow_events[i],
                "nearest_references": policy.nearest_refs[i],
                "final_position_error": float(final_position[i]),
                "final_angle_error": float(final_angle[i]),
                "task_sha256": task["task_sha256"],
                "input_sha256": input_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "action_array_key": action_key,
                "action_array_sha256": action_sha,
                "started_at": started[i],
                "ended_at": ended,
            }
        )
    return records, action_arrays


def merge_evaluation_shards(
    jsonl_shards: Sequence[Path], action_shards: Sequence[Path], output_jsonl: Path, output_actions: Path
) -> None:
    """Finalize disjoint group shards into the two registered raw artifacts."""
    if len(jsonl_shards) != len(action_shards) or not jsonl_shards:
        raise ValueError("rollout and action shard lists must be nonempty and aligned")
    if output_jsonl.exists() or output_actions.exists():
        raise FileExistsError("refusing to overwrite finalized raw artifacts")
    records: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for json_path, action_path in zip(jsonl_shards, action_shards):
        records.extend(line for line in json_path.read_text(encoding="utf-8").splitlines() if line)
        with np.load(action_path, allow_pickle=False) as shard:
            for key in shard.files:
                if key in arrays:
                    raise ValueError(f"duplicate raw action key: {key}")
                arrays[key] = np.asarray(shard[key])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_actions.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("".join(line + "\n" for line in records), encoding="utf-8")
    np.savez(output_actions, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--d", type=int, choices=tuple(BUDGETS), required=True)
    parser.add_argument("--model-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--evaluation-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--nearest-bank", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mac-preflight", action="store_true")
    parser.add_argument("--stage2-shard", action="store_true")
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_seed != args.evaluation_seed:
        raise ValueError("each registered model seed uses its identically seeded 50-task group")
    if args.mac_preflight and (args.model_seed != 42 or args.condition == "nearest_demo"):
        raise ValueError("Mac preflight uses seed 42 and the four trained-policy conditions")
    if args.stage2_shard:
        if args.mac_preflight or not args.checkpoint.is_file():
            raise ValueError("a full-GPU stage-two shard requires its completed checkpoint")
    else:
        checkpoint_set_manifest(args.checkpoint, mac_preflight=args.mac_preflight)
    if not args.mac_preflight:
        write_environment_once(
            args.artifact_root / "environment_manifest_gpu_evaluate.json",
            {
                "profile": "gpu_evaluate",
                "platform": platform.platform(),
                "python": sys.version,
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "device": args.device,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            },
        )
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    expected_update = 500 if args.mac_preflight else 30_000
    if metadata["update"] != expected_update:
        raise ValueError("checkpoint is not a registered final update")
    tasks = load_tasks(args.tasks, args.d, args.evaluation_seed, count=2 if args.mac_preflight else 50)
    nearest = None
    if args.condition == "nearest_demo":
        if args.nearest_bank is None:
            raise ValueError("nearest-demo evaluation requires a bank")
        with np.load(args.nearest_bank, allow_pickle=False) as bank:
            nearest = NearestDemonstration(**{key: bank[key] for key in bank.files})
    records, actions = run_group(
        model=model,
        condition=args.condition,
        tasks=tasks,
        hdf5=args.hdf5,
        normalizers=Normalizers.load(args.normalizers),
        checkpoint_sha256=sha256_file(args.checkpoint),
        input_sha256=sha256_file(args.hdf5),
        device=torch.device(args.device),
        nearest=nearest,
    )
    if args.output_jsonl.exists() or args.output_actions.exists():
        raise FileExistsError("refusing to overwrite evaluation outputs")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    np.savez(args.output_actions, **actions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
