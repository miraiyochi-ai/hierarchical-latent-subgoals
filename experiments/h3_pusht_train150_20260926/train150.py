#!/usr/bin/env python3
"""Extend only H3's training goal horizons; reuse the unchanged trainer/model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXPERIMENT_ID = "h3_pusht_train150_20260926"
HORIZONS = (25, 50, 75, 100, 150)
SOURCE_HASHES = {
    "a2a_h3_state_subgoal_pusht/h3_data.py": "173bfc47a7f9221317be9f4cef553d95a88794533bf98f28f44b94ef65b2c9a4",
    "a2a_h3_state_subgoal_pusht/h3_train.py": "32ce57323dfca9e04d2c37302d1930781745d1fd49fc37d7d688a8e5aad7a2a4",
    # Public release: h3_model.py differs from the run-time original only in the default
    # LeWM code/checkpoint locations (original sha256 45d67bcc487f53c01802e6cb9793f5d43690b4f01c2d135c8f49ba12944c1820).
    "a2a_h3_state_subgoal_pusht/h3_model.py": "8419e2bedc770e561c5e542bf49f9ab5d5bdea200e9542be24a2de41634bc6d7",
    "a2a_h1_clockwork_pusht_gate/h1_data.py": "34e4628c4a92c630d6cec11b237d95b2ab5a6686bcc93ec8a7809ce43e38b768",
    "a2a_h1_clockwork_pusht_gate/h1_train.py": "2c9991d340dfdfdb5031b3b6cdba2a03be504ffaaab512afb80ac032c1cb8dbe",
    "a2a_h1_clockwork_pusht_gate/h1_model.py": "e5038b96d3b16bcf978720c03c16a27b3051e1fa96eaf32079359537f6fa491a",
}
INPUT_HASHES = {
    "split_ids.json": "9fb8235254654cb8e8ebf481809fe0084bcb31afaacdae009b2a79627e3afc35",
    "normalizers.npz": "105c53f553f87a06f538bc259ca0f9fc9a79875d051c0bb704f5e7cfc70ccf25",
    "s2_normalizer.npz": "b146a404510e309346257ab8e8583c049cd8066d5b3b701959443ed39f9a5822",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    actual = {name: digest(root / name) for name in expected}
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise RuntimeError(f"train150 input/source hash mismatch: {changed}")
    return actual


def balanced_horizon_counts(update: int, batch_size: int = 128) -> dict[int, int]:
    """Same rotating-remainder rule as H3, with a five-update cycle."""
    if update < 0 or batch_size <= 0:
        raise ValueError("nonnegative update and positive batch_size required")
    base, remainder = divmod(batch_size, len(HORIZONS))
    counts = {d: base for d in HORIZONS}
    offset = update % len(HORIZONS)
    order = HORIZONS[offset:] + HORIZONS[:offset]
    for d in order[:remainder]:
        counts[d] += 1
    return counts


def install_horizons(data_module) -> None:
    if tuple(data_module.HORIZONS) != (25, 50, 75):
        raise RuntimeError("expected untouched three-horizon H3 data module")
    data_module.HORIZONS = HORIZONS
    data_module.balanced_horizon_counts = balanced_horizon_counts


def pool_sizes(lengths, ids) -> dict[str, dict[str, int]]:
    """Count the original enumeration exactly, without allocating its objects."""
    return {
        str(d): {
            "task_starts": sum(max(0, int(lengths[i]) - d) for i in ids),
            "l1_specs": sum(max(0, int(lengths[i]) - d) for i in ids) * len(range(0, d - 8 + 1, 8)),
            "l2_specs": sum(max(0, int(lengths[i]) - d) for i in ids) * len(range(0, d, 16)),
        }
        for d in HORIZONS
    }


def extended_protocol(original: dict, profile: str = "gpu_full") -> dict:
    expected = {"horizons": [25, 50, 75], "updates": 30000,
                "model_seeds": [42, 43, 44]}
    if any(original.get(key) != value for key, value in expected.items()):
        raise RuntimeError("unexpected upstream protocol; review instead of rewriting")
    if profile not in ("gpu_full", "mac_preflight"):
        raise ValueError("unsupported profile")
    smoke = profile == "mac_preflight"
    updates, batch_size = (8, 8) if smoke else (30000, 128)
    counts = {d: 0 for d in HORIZONS}
    for update in range(updates):
        for d, count in balanced_horizon_counts(update, batch_size).items():
            counts[d] += count
    return {
        **original,
        "experiment": EXPERIMENT_ID,
        "source_experiment": "a2a_h3_state_subgoal_pusht",
        "horizons": list(HORIZONS),
        "model_seeds": [42],
        "profile": profile,
        "updates": updates,
        "batch": {"layer1": batch_size, "layer2": batch_size},
        "microbatch_size": 2 if smoke else 32,
        "training_conditions": ["hierarchical", "flat-direct-goal"],
        "initialization": "from_scratch_not_finetuning",
        "horizon_allocation": "equal base + rotating remainder; cycle 5 updates",
        "allocated_samples_per_horizon_per_layer": counts,
        "only_algorithm_change": "training goal horizons and corresponding balanced remainder cycle",
        "evaluation_note": "conditions is the upstream supported-condition list, not a claim of completed evaluation",
    }


def install_snapshot_hook(trainer, model_module, output_root: Path, interval: int) -> None:
    """Save evaluation-only EMA snapshots; do not pretend they can resume training."""
    if interval == 0:
        return
    original_ema = trainer.PowerEMA

    class SnapshotEMA(original_ema):
        def update(self, model, update):
            decay = super().update(model, update)
            completed = update + 1
            if completed % interval == 0 and completed < 30000:
                kind = "hierarchical" if model.config.hierarchical else "flat_direct_goal"
                path = output_root / "ema_snapshots" / f"{kind}_seed42_update{completed:05d}.pt"
                partial = path.with_suffix(".partial.pt")
                if path.exists() or partial.exists():
                    raise FileExistsError(f"refusing to overwrite snapshot: {path}")
                model_module.save_checkpoint(partial, model, completed, self.state)
                payload = trainer.torch.load(partial, map_location="cpu", weights_only=False)
                if payload["update"] != completed or set(payload["ema_state"]) != set(self.state):
                    raise RuntimeError("snapshot metadata verification failed")
                for key, value in self.state.items():
                    if not trainer.torch.equal(payload["ema_state"][key], value.detach().cpu()):
                        raise RuntimeError(f"snapshot tensor verification failed: {key}")
                partial.rename(path)
                trainer._write_identical_once(path.with_suffix(".json"), {
                    "update": completed, "sha256": digest(path),
                    "purpose": "evaluation_only_not_resumable",
                    "not_saved": ["optimizer", "rng", "data_sampler_cursor"],
                    "final_evaluation_checkpoint": False,
                })
            return decay

    trainer.PowerEMA = SnapshotEMA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("gpu_full", "mac_preflight"), default="gpu_full")
    parser.add_argument("--condition", choices=("hierarchical", "flat-direct-goal"), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--supply", choices=("reference", "optimized"), default="optimized")
    parser.add_argument("--data-workers", type=int)
    parser.add_argument("--snapshot-every", type=int, choices=(0, 5000), default=5000,
                        help="Evaluation-only EMA snapshots; never resumable training state")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify inputs/counts on CPU; do not construct models or train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = verify_hashes(EXPERIMENTS, SOURCE_HASHES)
    inputs = verify_hashes(args.data_root, INPUT_HASHES)
    if args.hdf5.stat().st_size != 46300921856:
        raise RuntimeError("unexpected uncompressed HDF5 size")
    # This runner cannot resume. Never silently append a new run to old losses.
    for forbidden in ("training_losses.jsonl", "checkpoints", "ema_snapshots",
                      "g3_mac_preflight/raw_losses.jsonl", "g3_mac_preflight/checkpoints"):
        if (args.output_root / forbidden).exists():
            raise FileExistsError(f"fresh output required: {args.output_root / forbidden}")
    sys.path.insert(0, str(EXPERIMENTS / "a2a_h3_state_subgoal_pusht"))
    import h3_data
    import h3_model
    import h3_train

    verify_hashes(h3_model.LEWM_CHECKPOINT.parent,
                  {h3_model.LEWM_CHECKPOINT.name: h3_model.LEWM_SHA256})
    splits = h3_data.load_split_ids(args.data_root / "split_ids.json")
    table = h3_data.EpisodeTable.from_hdf5(args.hdf5)
    smoke = args.profile == "mac_preflight"
    ids = splits["train"][:64] if smoke else splits["train"]
    sizes = pool_sizes(table.lengths, ids)
    if any(cell["task_starts"] <= 0 for cell in sizes.values()):
        raise RuntimeError("one or more training horizon pools are empty")
    manifest = {
        "experiment": EXPERIMENT_ID, "condition": args.condition, "seed": 42,
        "training_horizons": list(HORIZONS), "updates": 8 if smoke else 30000,
        "profile": args.profile, "batch_size": 8 if smoke else 128,
        "microbatch_size": 2 if smoke else 32,
        "initialization": "from_scratch_not_finetuning",
        "upstream_source_sha256": sources, "input_sha256": inputs,
        "adapter_sha256": digest(Path(__file__).resolve()),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "training_episode_count_used": len(ids),
        "train_pool_sizes": sizes,
        "checkpoint_format_note": "Original H3 v3 format retained for unchanged evaluators; this manifest identifies the new training distribution",
        "checkpoint_policy": "Original final EMA for evaluation; optional intermediate EMA snapshots are not resumable",
        "snapshot_every": 0 if smoke else args.snapshot_every,
        "claim_boundary": "seed42 single-seed diagnostic, not a new three-seed H3 gate pass",
    }
    original_writer = h3_train._write_identical_once
    original_writer(args.output_root / "train150_manifest.json", manifest)
    if args.preflight_only:
        print(json.dumps({"state": "PREFLIGHT_PASSED_NOT_TRAINED", **manifest}, indent=2))
        return 0
    install_horizons(h3_data)

    def protocol_writer(path, value):
        if path == args.output_root / "protocol.json":
            value = extended_protocol(value, args.profile)
        return original_writer(path, value)

    # Leave the optimizer, model construction, train_one, losses and checkpoint
    # serialization intact. Only inject validated arguments and truthful metadata.
    h3_train.parse_args = lambda: args
    h3_train._write_identical_once = protocol_writer
    install_snapshot_hook(h3_train, h3_model, args.output_root, 0 if smoke else args.snapshot_every)
    result = h3_train.main()
    verify_hashes(EXPERIMENTS, SOURCE_HASHES)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
