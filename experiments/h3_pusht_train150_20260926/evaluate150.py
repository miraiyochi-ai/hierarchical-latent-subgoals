#!/usr/bin/env python3
"""Run an unchanged H3/H3x evaluation cell with post-step progress reporting."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train150

EVALUATION_SOURCES = {
    **train150.SOURCE_HASHES,
    "a2a_h3_state_subgoal_pusht/h3_evaluate.py": "6a38532da76070534cacfd6026c78a51e5a14b59a274bfa05b35085bca3e88e4",
    # Public release: h3x_evaluate.py differs from the run-time original only in the default
    # dataset location (original sha256 66259cdcbb63add4d4c477c2e8939513c1f2b20b0c9cc8fe47ba907a71b73c56).
    "a2a_h3x_horizon_extension/h3x_evaluate.py": "a0dff4673e4a5ae83868fbfb1808464ae1bd2e737b5a34f0f8fa4e26826bdc16",
    "a2a_h1_clockwork_pusht_gate/h1_evaluate.py": "af72dde7324a1ceb7303a101c1ff838a4775c4b77e718c80b7b5f9c5075560fe",
}
TASK_HASHES = {
    "evaluation_tasks.jsonl": "6b347a6416000e8b35352ef72594b71d6e78d9a833edb7c91e393afcdfe7b318",
    "evaluation_tasks_extended.jsonl": "ac1f0066fcc8de283ed36b4cadeb0b4588b8fd7389b93ab8fb7870cb0de77961",
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


class Progress:
    """Count actually executed environment steps, including terminal steps.

    The original observe receives the *post-step* active mask, so its own steps
    omit the final step of newly terminated episodes. We never change that state.
    The previous mask identifies exactly which environments just executed a step.
    """

    def __init__(self, metadata: dict, writer, *, interval: float = 5.0, clock=time.monotonic):
        self.metadata = dict(metadata)
        self.writer = writer
        self.interval = interval
        self.clock = clock
        self.previous_active = [True] * 50
        self.cursor = 0
        self.loop_steps = 0
        self.last_write = -math.inf
        self.started = time.time()

    def report(self, phase: str, state: str, **extra) -> None:
        self.writer({
            **self.metadata,
            "state": state, "phase": phase, "cursor": self.cursor,
            "cursor_kind": "executed_environment_steps",
            "loop_steps": self.loop_steps, "expected_episodes": 50,
            "episodes_inactive_before_budget_end": 50 - sum(self.previous_active),
            "active_episodes": sum(self.previous_active),
            "pid": os.getpid(), "started_at": self.started,
            "observed_at": time.time(), **extra,
        })
        self.last_write = self.clock()

    def after_observe(self, active) -> None:
        current = [bool(value) for value in active]
        if len(current) != 50:
            raise RuntimeError("progress observer expected exactly 50 tasks")
        if any(now and not before for now, before in zip(current, self.previous_active)):
            raise RuntimeError("unexpected environment reactivation")
        self.cursor += sum(self.previous_active)
        self.loop_steps += 1
        changed = current != self.previous_active
        self.previous_active = current
        if changed or self.loop_steps == 1 or self.clock() - self.last_write >= self.interval:
            self.report("closed_loop_evaluation", "PROGRESSING")


@contextmanager
def observe_progress(policy_class, progress: Progress):
    original = policy_class.observe

    def wrapped(self, images, agents, active):
        result = original(self, images, agents, active)
        progress.after_observe(active)
        return result

    policy_class.observe = wrapped
    try:
        yield
    finally:
        policy_class.observe = original


def select_tasks(path: Path, d: int) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line]
    selected = [task for task in tasks if int(task["d"]) == d and int(task["seed"]) == 42]
    if len(selected) != 50:
        raise RuntimeError(f"expected 50 fixed tasks, got {len(selected)}")
    keys = [(int(task["episode_idx"]), int(task["start_step"])) for task in selected]
    if len(set(keys)) != 50:
        raise RuntimeError("duplicate fixed task identifiers")
    if any(int(task["goal_step"]) - int(task["start_step"]) != d or int(task["eval_budget"]) != 2 * d for task in selected):
        raise RuntimeError("unexpected task horizon/budget")
    return sorted(selected, key=lambda task: (int(task["episode_idx"]), int(task["start_step"])))


def stage_identical(source: Path, destination: Path, expected: str) -> None:
    if train150.digest(source) != expected:
        raise RuntimeError(f"source hash mismatch: {source}")
    if destination.exists():
        if train150.digest(destination) != expected:
            raise RuntimeError(f"refusing to replace conflicting staged input: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents accidental overwrites by concurrent cells.
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    except FileExistsError:
        if train150.digest(destination) != expected:
            raise RuntimeError(f"concurrent staged input is incomplete: {destination}")
    if train150.digest(destination) != expected:
        raise RuntimeError(f"staged input verification failed: {destination}")


def validate_rows(rows: list[dict], tasks: list[dict], condition: str, d: int,
                  checkpoint_sha: str, task_file_sha: str) -> dict:
    if len(rows) != 50:
        raise RuntimeError(f"expected 50 result rows, got {len(rows)}")
    expected = {(int(task["episode_idx"]), int(task["start_step"])): task for task in tasks}
    observed = set()
    for row in rows:
        key = (int(row["episode_idx"]), int(row["start_step"]))
        if key not in expected or key in observed:
            raise RuntimeError("result task membership/uniqueness mismatch")
        observed.add(key)
        task = expected[key]
        checks = {
            "condition": condition, "d": d, "model_seed": 42, "evaluation_seed": 42,
            "goal_step": int(task["goal_step"]), "task_sha256": task["task_sha256"],
            "checkpoint_sha256": checkpoint_sha, "input_sha256": task_file_sha,
        }
        if any(row.get(name) != value for name, value in checks.items()):
            raise RuntimeError(f"result provenance mismatch: {key}")
        if not isinstance(row.get("success"), bool):
            raise RuntimeError("success must be a boolean")
        if not (1 <= int(row["executed_steps"]) <= 2 * d):
            raise RuntimeError("invalid executed step count")
        if not all(math.isfinite(float(row[name])) for name in ("final_position_error", "final_angle_error")):
            raise RuntimeError("non-finite final error")
        if row["success"]:
            if row.get("success_step") != row["executed_steps"] or row.get("termination_reason") != "environment_terminated":
                raise RuntimeError("inconsistent success termination record")
        elif row.get("success_step") is not None:
            raise RuntimeError("failed episode has success step")
    return {
        "episodes": 50,
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / 50,
        "executed_environment_steps": sum(int(row["executed_steps"]) for row in rows),
    }


def validate_arrays(path: Path, rows: list[dict]) -> None:
    import hashlib
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {row["action_array_key"] for row in rows}:
            raise RuntimeError("action array membership mismatch")
        for row in rows:
            value = arrays[row["action_array_key"]]
            if value.shape != (int(row["executed_steps"]), 2) or not np.isfinite(value).all():
                raise RuntimeError("invalid action array shape/values")
            sha = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
            if sha != row["action_array_sha256"]:
                raise RuntimeError("action array hash mismatch")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--condition", choices=("predicted", "flat-direct-goal"), required=True)
    parser.add_argument("--d", type=int, choices=train150.HORIZONS, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_interval < 0:
        raise ValueError("progress interval cannot be negative")
    model_root = args.model_root.resolve()
    status_path = model_root / f"evaluation_progress_{args.condition}_d{args.d}.json"
    complete_path = model_root / f"evaluation_complete_{args.condition}_d{args.d}.json"
    if status_path.exists() or complete_path.exists():
        raise FileExistsError("evaluation cell already has status; inspect rather than rerun")
    output_root = model_root / ("evaluation_shards" if args.d <= 75 else "h3x")
    stem = f"raw_rollouts_{args.condition}_d{args.d}_m42_e42"
    outputs = [output_root / f"{stem}.jsonl", output_root / f"{stem}.npz",
               output_root / f"{stem}_instrumentation.npz"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite existing evaluation outputs")
    kind = "hierarchical" if args.condition == "predicted" else "flat_direct_goal"
    checkpoint = model_root / "checkpoints/seed_42" / f"{kind}_ema.pt"
    task_name = "evaluation_tasks.jsonl" if args.d <= 75 else "evaluation_tasks_extended.jsonl"
    task_source = args.data_root / task_name
    metadata = {
        "experiment": train150.EXPERIMENT_ID, "condition": args.condition,
        "d": args.d, "model_seed": 42, "evaluation_seed": 42,
        "checkpoint": str(checkpoint), "output_root": str(output_root),
        "progress_path": str(status_path), "completion_path": str(complete_path),
        "adapter_sha256": train150.digest(Path(__file__).resolve()),
    }
    progress = Progress(metadata, lambda value: atomic_json(status_path, value),
                        interval=args.progress_interval)
    progress.report("input_verification", "STARTING")
    old_argv = sys.argv
    environment = {
        "H3X_H3_RESULTS": str(model_root), "H3X_OUT": str(model_root / "h3x"),
        "H3X_H5": str(args.hdf5.resolve()),
    }
    previous_environment = {key: os.environ.get(key) for key in environment}
    try:
        train150.verify_hashes(train150.EXPERIMENTS, EVALUATION_SOURCES)
        train150.verify_hashes(args.data_root, {**train150.INPUT_HASHES, task_name: TASK_HASHES[task_name]})
        if args.hdf5.stat().st_size != 46300921856:
            raise RuntimeError("unexpected HDF5 size")
        manifest = json.loads((model_root / "train150_manifest.json").read_text())
        expected_condition = "hierarchical" if args.condition == "predicted" else "flat-direct-goal"
        if (manifest.get("experiment"), manifest.get("condition"), manifest.get("seed"), manifest.get("updates")) != (train150.EXPERIMENT_ID, expected_condition, 42, 30000):
            raise RuntimeError("not a matching train150 full-training checkpoint root")
        tasks = select_tasks(task_source, args.d)
        checkpoint_sha = train150.digest(checkpoint)
        for name in ("normalizers.npz", "s2_normalizer.npz"):
            stage_identical(args.data_root / name, model_root / "data" / name, train150.INPUT_HASHES[name])
        if args.d >= 100:
            stage_identical(task_source, output_root / "data" / task_name, TASK_HASHES[task_name])
        os.environ.update(environment)
        sys.path.insert(0, str(train150.EXPERIMENTS / "a2a_h3_state_subgoal_pusht"))
        import h3_evaluate
        progress.metadata.update(checkpoint_sha256=checkpoint_sha, input_sha256=TASK_HASHES[task_name], h3x_environment=environment)
        progress.report("checkpoint_and_environment_loading", "STARTING")
        with observe_progress(h3_evaluate.H3ClockedBatchPolicy, progress):
            if args.d <= 75:
                original_parse_args = h3_evaluate.parse_args
                h3_evaluate.parse_args = lambda: argparse.Namespace(
                    checkpoint=checkpoint, condition=args.condition, d=args.d,
                    evaluation_seed=42, hdf5=args.hdf5, data_root=args.data_root,
                    output_root=output_root, device=args.device, mac_preflight=False)
                try:
                    exit_code = h3_evaluate.main()
                finally:
                    h3_evaluate.parse_args = original_parse_args
            else:
                sys.path.insert(0, str(train150.EXPERIMENTS / "a2a_h3x_horizon_extension"))
                import h3x_evaluate
                # Explicit paths, including when an embedding interpreter imported
                # the original module before this wrapper installed environment.
                h3x_evaluate.H3_RESULTS = model_root
                h3x_evaluate.OUT_ROOT = output_root
                h3x_evaluate.H5 = args.hdf5
                sys.argv = ["h3x_evaluate.py", "--condition", args.condition,
                            "--d", str(args.d), "--seed", "42", "--device", args.device]
                exit_code = h3x_evaluate.main()
        if exit_code != 0:
            raise RuntimeError(f"upstream evaluation returned {exit_code}")
        progress.report("output_verification", "VERIFYING")
        rows = [json.loads(line) for line in outputs[0].read_text().splitlines() if line]
        summary = validate_rows(rows, tasks, args.condition, args.d, checkpoint_sha, TASK_HASHES[task_name])
        validate_arrays(outputs[1], rows)
        if summary["executed_environment_steps"] != progress.cursor:
            raise RuntimeError("post-step progress cursor disagrees with final executed steps")
        if train150.digest(checkpoint) != checkpoint_sha:
            raise RuntimeError("evaluation checkpoint changed")
        train150.verify_hashes(train150.EXPERIMENTS, EVALUATION_SOURCES)
        artifact_hashes = {str(path.relative_to(model_root)): train150.digest(path) for path in outputs}
        completion = {
            **progress.metadata, **summary, "state": "COMPLETED",
            "phase": "complete", "cursor": progress.cursor,
            "cursor_kind": "executed_environment_steps", "exit_code": 0,
            "completed_at": time.time(), "artifact_sha256": artifact_hashes,
            "source_sha256": EVALUATION_SOURCES,
            "claim_boundary": "one evaluation cell; not completion of the whole comparison",
        }
        atomic_json(complete_path, completion)
        progress.report("complete", "COMPLETED", verified_episodes=50,
                        successes=summary["successes"], success_rate=summary["success_rate"],
                        completion_sha256=train150.digest(complete_path))
        print(json.dumps(completion, indent=2))
        return 0
    except BaseException as error:
        progress.report("failed", "FAILED", error=repr(error), error_type=type(error).__name__)
        raise
    finally:
        sys.argv = old_argv
        for key, old in previous_environment.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


if __name__ == "__main__":
    raise SystemExit(main())
