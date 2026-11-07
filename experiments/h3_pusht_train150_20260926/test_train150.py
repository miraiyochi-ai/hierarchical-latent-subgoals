"""CPU contract tests: exact upstream sampler definitions, no model/GPU/data."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train150


def source_definitions(path, names, namespace):
    """Execute selected, unmodified definitions without heavyweight imports."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("missing upstream definition")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def data_module():
    module = types.ModuleType("train150_contract_data")
    sys.modules[module.__name__] = module
    module.__dict__.update(dataclass=dataclass, np=np, HORIZONS=(25, 50, 75),
                           FULL_BATCH_SIZE=128, L1_CHUNK=8, L2_CLOCK=16, L2_WINDOW=16)
    source_definitions(train150.EXPERIMENTS / "a2a_h1_clockwork_pusht_gate/h1_data.py",
                       ["CyclicSpecSampler"], module.__dict__)
    source_definitions(train150.EXPERIMENTS / "a2a_h3_state_subgoal_pusht/h3_data.py",
                       ["Layer1Spec", "Layer2Spec", "_valid_task_starts",
                        "enumerate_layer1_specs", "enumerate_layer2_specs",
                        "balanced_horizon_counts", "BoundarySampler",
                        "layer2_reference_steps", "validate_no_leak"], module.__dict__)
    return module


class Train150Contracts(unittest.TestCase):
    def test_upstream_sources_unchanged(self):
        self.assertEqual(train150.verify_hashes(train150.EXPERIMENTS, train150.SOURCE_HASHES), train150.SOURCE_HASHES)

    def test_exact_five_update_balance(self):
        totals = Counter()
        for update in range(5):
            counts = train150.balanced_horizon_counts(update)
            self.assertEqual(sum(counts.values()), 128)
            self.assertEqual(sorted(counts.values()), [25, 25, 26, 26, 26])
            totals.update(counts)
        self.assertEqual(dict(totals), {d: 128 for d in train150.HORIZONS})

    def test_full_run_equal_horizon_counts(self):
        totals = Counter()
        for update in range(30000):
            totals.update(train150.balanced_horizon_counts(update))
        self.assertEqual(dict(totals), {d: 768000 for d in train150.HORIZONS})

    def test_cycle_and_small_batches(self):
        for size in (1, 2, 5, 8, 32, 128):
            totals = Counter()
            for update in range(5):
                counts = train150.balanced_horizon_counts(update, size)
                self.assertEqual(counts, train150.balanced_horizon_counts(update + 5, size))
                self.assertEqual(sum(counts.values()), size)
                totals.update(counts)
            self.assertEqual(dict(totals), {d: size for d in train150.HORIZONS})

    def test_old_sampler_exposes_mod_three_bias(self):
        data = data_module()
        data.HORIZONS = train150.HORIZONS
        totals = Counter()
        for update in range(15):
            totals.update(data.balanced_horizon_counts(update))
        self.assertGreater(len(set(totals.values())), 1)

    def test_short_pool_membership_unchanged_and_no_leaks(self):
        data = data_module()
        table = types.SimpleNamespace(lengths=[171, 162, 155])
        # Deliberately omit episode 1, representing held-out data.
        ids = [0, 2]
        old_l1 = data.enumerate_layer1_specs(table, ids)
        old_l2 = data.enumerate_layer2_specs(table, ids)
        train150.install_horizons(data)
        l1 = data.enumerate_layer1_specs(table, ids)
        l2 = data.enumerate_layer2_specs(table, ids)
        for d in (25, 50, 75):
            self.assertEqual(l1[d], old_l1[d])
        for key in old_l2:
            self.assertEqual(l2[key], old_l2[key])
        first = [spec for pool in l1.values() for spec in pool]
        second = [spec for pool in l2.values() for spec in pool]
        self.assertEqual(set(spec.episode_idx for spec in first + second), {0, 2})
        self.assertTrue(all(value == 0 for value in data.validate_no_leak(first, second).values()))
        for spec in second:
            self.assertLess(spec.goal_step, table.lengths[spec.episode_idx])
            self.assertEqual(spec.teacher_step, min(spec.task_start_step + spec.boundary + 16, spec.goal_step))
        sizes = train150.pool_sizes(table.lengths, ids)
        for d in train150.HORIZONS:
            self.assertEqual(sizes[str(d)]["l1_specs"], len(l1[d]))
            self.assertEqual(sizes[str(d)]["l2_specs"], sum(len(pool) for (dd, _), pool in l2.items() if dd == d))

    def test_boundary_cycle_and_hier_flat_batch_pairing(self):
        data = data_module()
        train150.install_horizons(data)
        table = types.SimpleNamespace(lengths=[160])
        l1 = data.enumerate_layer1_specs(table, [0])
        l2 = data.enumerate_layer2_specs(table, [0])
        hier = data.CyclicSpecSampler(l1, 42)
        flat = data.CyclicSpecSampler(l1, 42)
        upper = data.BoundarySampler(l2, 42)
        upper_again = data.BoundarySampler(l2, 42)
        boundary_counts = {d: Counter() for d in train150.HORIZONS}
        for update in range(25):
            for d, count in train150.balanced_horizon_counts(update).items():
                self.assertEqual(hier.take(d, count), flat.take(d, count))
            specs = upper.take_update(update)
            self.assertEqual(specs, upper_again.take_update(update))
            self.assertEqual(Counter(spec.d for spec in specs), train150.balanced_horizon_counts(update))
            for spec in specs:
                boundary_counts[spec.d][spec.boundary] += 1
        for d, counts in boundary_counts.items():
            self.assertEqual(set(counts), set(range(0, d, 16)))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_protocol_truthful_and_no_accidental_change(self):
        old = dict(horizons=[25, 50, 75], updates=30000, model_seeds=[42, 43, 44],
                   clock=16, prediction_horizon=16, hold=16, layer1_chunk=8,
                   euler_steps=6, batch={"layer1": 128, "layer2": 128})
        new = train150.extended_protocol(old)
        self.assertEqual(new["horizons"], [25, 50, 75, 100, 150])
        self.assertEqual(new["model_seeds"], [42])
        for key in old.keys() - {"horizons", "model_seeds"}:
            self.assertEqual(new[key], old[key])
        self.assertEqual(old["horizons"], [25, 50, 75])
        with self.assertRaises(RuntimeError):
            train150.extended_protocol({**old, "updates": 10000})
        smoke = train150.extended_protocol(old, "mac_preflight")
        self.assertEqual(smoke["updates"], 8)
        self.assertEqual(smoke["batch"], {"layer1": 8, "layer2": 8})
        self.assertEqual(smoke["microbatch_size"], 2)
        self.assertEqual(sum(smoke["allocated_samples_per_horizon_per_layer"].values()), 64)
        self.assertEqual(new["allocated_samples_per_horizon_per_layer"], {d: 768000 for d in train150.HORIZONS})

    def test_snapshot_hook_preserves_ema_return_and_full_final(self):
        class FakeEMA:
            def update(self, model, update):
                return update + 0.25
        trainer = types.SimpleNamespace(PowerEMA=FakeEMA)
        train150.install_snapshot_hook(trainer, None, Path("unused"), 0)
        self.assertIs(trainer.PowerEMA, FakeEMA)
        train150.install_snapshot_hook(trainer, None, Path("unused"), 5000)
        for update in (0, 4998, 5000, 29999):
            self.assertEqual(trainer.PowerEMA().update(None, update), update + 0.25)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable locally; run on provisioned CPU runtime")
    def test_snapshot_real_tensor_roundtrip_and_rng_unchanged(self):
        import torch
        save_module = types.SimpleNamespace()
        namespace = dict(torch=torch, asdict=asdict, hashlib=hashlib,
                         CHECKPOINT_VERSION=3, EXPERIMENT_ID="a2a_h3_state_subgoal_pusht")
        source_definitions(train150.EXPERIMENTS / "a2a_h3_state_subgoal_pusht/h3_model.py",
                           ["module_state_sha256", "save_checkpoint"], namespace)
        save_module.save_checkpoint = namespace["save_checkpoint"]

        @dataclass
        class Config:
            hierarchical: bool = True
            seed: int = 42

        model = torch.nn.Linear(2, 3)
        model.config = Config()
        model.lewm = torch.nn.Linear(2, 2)

        class TinyEMA:
            def __init__(self):
                self.state = {k: value.detach().clone() for k, value in model.state_dict().items()}
            def update(self, model, update):
                return 0.9

        trainer = types.SimpleNamespace(PowerEMA=TinyEMA, torch=torch)
        trainer._write_identical_once = lambda path, value: path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory(prefix="train150_snapshot_contract_") as directory:
            root = Path(directory)
            train150.install_snapshot_hook(trainer, save_module, root, 5000)
            before = torch.random.get_rng_state().clone()
            self.assertEqual(trainer.PowerEMA().update(model, 4999), 0.9)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            path = root / "ema_snapshots/hierarchical_seed42_update05000.pt"
            self.assertTrue(path.is_file())
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["update"], 5000)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(payload["ema_state"][name], value))
            summary = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(summary["sha256"], train150.digest(path))
            self.assertFalse(summary["final_evaluation_checkpoint"])
            self.assertEqual(summary["purpose"], "evaluation_only_not_resumable")
            with self.assertRaises(FileExistsError):
                trainer.PowerEMA().update(model, 4999)

    def test_install_requires_original_module(self):
        data = data_module()
        train150.install_horizons(data)
        with self.assertRaises(RuntimeError):
            train150.install_horizons(data)


if __name__ == "__main__":
    unittest.main()
