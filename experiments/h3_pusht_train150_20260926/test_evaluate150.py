"""CPU tests for post-environment-step monitoring, without models or GPU."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate150


class EvaluationContracts(unittest.TestCase):
    def test_sources_unchanged(self):
        self.assertEqual(evaluate150.train150.verify_hashes(evaluate150.train150.EXPERIMENTS, evaluate150.EVALUATION_SOURCES), evaluate150.EVALUATION_SOURCES)

    def test_progress_counts_terminal_steps_not_policy_clock(self):
        reports = []
        progress = evaluate150.Progress({}, reports.append, interval=0)
        progress.report("setup", "STARTING")
        progress.after_observe([False] * 3 + [True] * 47)
        self.assertEqual(progress.cursor, 50)
        self.assertEqual(reports[-1]["active_episodes"], 47)
        progress.after_observe([False] * 10 + [True] * 40)
        self.assertEqual(progress.cursor, 97)
        progress.after_observe([False] * 50)
        self.assertEqual(progress.cursor, 137)
        self.assertEqual(progress.loop_steps, 3)
        self.assertEqual(reports[-1]["episodes_inactive_before_budget_end"], 50)

    def test_reactivation_and_wrong_batch_rejected(self):
        progress = evaluate150.Progress({}, lambda value: None)
        with self.assertRaises(RuntimeError):
            progress.after_observe([True])
        progress.after_observe([False] + [True] * 49)
        with self.assertRaises(RuntimeError):
            progress.after_observe([True] * 50)

    def test_throttling_does_not_lose_internal_steps(self):
        reports = []
        clock = [0.0]
        progress = evaluate150.Progress({}, reports.append, interval=5, clock=lambda: clock[0])
        progress.after_observe([True] * 50)
        for _ in range(4):
            progress.after_observe([True] * 50)
        self.assertEqual(len(reports), 1)
        self.assertEqual(progress.cursor, 250)
        clock[0] = 5.1
        progress.after_observe([True] * 50)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[-1]["cursor"], 300)

    def test_exact_original_observe_is_preserved_and_restored(self):
        path = evaluate150.train150.EXPERIMENTS / "a2a_h3_state_subgoal_pusht/h3_evaluate.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "H3ClockedBatchPolicy")
        observe = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "observe")
        code = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), observe], type_ignores=[]))
        namespace = {}
        exec(compile(code, str(path), "exec"), namespace)
        original = namespace["observe"]
        policy_class = type("Policy", (), {"observe": original})

        def policy():
            instance = policy_class()
            instance.tasks = [{} for _ in range(50)]
            instance.image_history = [[np.zeros((1,), np.uint8)] * 16 for _ in range(50)]
            instance.agent_history = [[np.zeros((2,), np.float32)] * 8 for _ in range(50)]
            instance.steps = np.zeros(50, np.int64)
            return instance

        baseline, monitored = policy(), policy()
        images = np.arange(50, dtype=np.uint8).reshape(50, 1)
        agents = np.arange(100, dtype=np.float32).reshape(50, 2)
        active = np.asarray([False] * 3 + [True] * 47)
        original(baseline, images, agents, active)
        progress = evaluate150.Progress({}, lambda value: None)
        before_rng = np.random.get_state()
        with evaluate150.observe_progress(policy_class, progress):
            self.assertIsNone(monitored.observe(images, agents, active))
        self.assertIs(policy_class.observe, original)
        self.assertTrue(np.array_equal(baseline.steps, monitored.steps))
        for field in ("image_history", "agent_history"):
            for first, second in zip(getattr(baseline, field), getattr(monitored, field)):
                self.assertTrue(np.array_equal(first, second))
        self.assertEqual(progress.cursor, 50)
        self.assertEqual(monitored.steps.sum(), 47)
        after_rng = np.random.get_state()
        self.assertEqual(before_rng[0], after_rng[0])
        self.assertTrue(np.array_equal(before_rng[1], after_rng[1]))
        self.assertEqual(before_rng[2:], after_rng[2:])
        with self.assertRaises(ValueError):
            with evaluate150.observe_progress(policy_class, progress):
                raise ValueError("test restoration")
        self.assertIs(policy_class.observe, original)

    def test_stage_input_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="evaluate150_input_") as directory:
            root = Path(directory)
            source, dest = root / "source", root / "dest"
            source.write_bytes(b"fixed")
            sha = evaluate150.train150.digest(source)
            evaluate150.stage_identical(source, dest, sha)
            self.assertEqual(dest.read_bytes(), b"fixed")
            evaluate150.stage_identical(source, dest, sha)
            dest.write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                evaluate150.stage_identical(source, dest, sha)
            self.assertEqual(dest.read_bytes(), b"different")

    def test_rows_and_arrays_have_exact_provenance(self):
        tasks = [dict(episode_idx=i, start_step=3, goal_step=103, d=100, seed=42,
                      eval_budget=200, task_sha256=f"task{i}") for i in range(50)]
        arrays = {f"action{i}": np.ones((2, 2), np.float32) for i in range(50)}
        rows = [dict(condition="predicted", d=100, model_seed=42, evaluation_seed=42,
                     episode_idx=i, start_step=3, goal_step=103, task_sha256=f"task{i}",
                     checkpoint_sha256="checkpoint", input_sha256="input", success=True,
                     executed_steps=2, success_step=2, termination_reason="environment_terminated",
                     final_position_error=1.0, final_angle_error=0.01, action_array_key=f"action{i}",
                     action_array_sha256=hashlib.sha256(arrays[f"action{i}"].view(np.uint8)).hexdigest()) for i in range(50)]
        summary = evaluate150.validate_rows(rows, tasks, "predicted", 100, "checkpoint", "input")
        self.assertEqual(summary, dict(episodes=50, successes=50, success_rate=1.0, executed_environment_steps=100))
        with tempfile.TemporaryDirectory(prefix="evaluate150_arrays_") as directory:
            path = Path(directory) / "actions.npz"
            np.savez_compressed(path, **arrays)
            evaluate150.validate_arrays(path, rows)
            rows[0]["action_array_sha256"] = "bad"
            with self.assertRaises(RuntimeError):
                evaluate150.validate_arrays(path, rows)
        rows[0]["model_seed"] = 43
        with self.assertRaises(RuntimeError):
            evaluate150.validate_rows(rows, tasks, "predicted", 100, "checkpoint", "input")

    def test_fixed_task_selection_and_duplicate_rejection(self):
        tasks = [dict(episode_idx=i, start_step=0, goal_step=150, d=150,
                      seed=42, eval_budget=300) for i in reversed(range(50))]
        with tempfile.TemporaryDirectory(prefix="evaluate150_tasks_") as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text("\n".join(json.dumps(task) for task in tasks))
            self.assertEqual([task["episode_idx"] for task in evaluate150.select_tasks(path, 150)], list(range(50)))
            tasks[0] = dict(tasks[1])
            path.write_text("\n".join(json.dumps(task) for task in tasks))
            with self.assertRaises(RuntimeError):
                evaluate150.select_tasks(path, 150)


if __name__ == "__main__":
    unittest.main()
