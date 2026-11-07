#!/usr/bin/env python3
"""H3x horizon-extension evaluation wrapper.

Reuses h3_evaluate's clock/condition machinery unchanged; only swaps the task
source (evaluation_tasks_extended.jsonl, d in {100,150}) and the output root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
H3_DIR = HERE.parents[0] / "a2a_h3_state_subgoal_pusht"
sys.path.insert(0, str(H3_DIR))

import h3_data  # noqa: E402
import h3_evaluate  # noqa: E402
import h3_model  # noqa: E402
import h1_evaluate as reused  # noqa: E402

import os  # noqa: E402

H3_RESULTS = Path(os.environ.get("H3X_H3_RESULTS", HERE.parents[1] / "results/a2a_h3_state_subgoal_pusht"))
OUT_ROOT = Path(os.environ.get("H3X_OUT", HERE.parents[1] / "results/a2a_h3x_horizon_extension"))
H5 = Path(os.environ.get("H3X_H5", HERE.parents[1] / "data/pusht/pusht_expert_train.h5"))
CONDITIONS = ("predicted", "oracle", "identity", "flat-direct-goal")


def load_tasks(d: int, seed: int) -> list[dict]:
    name = "evaluation_tasks_d200.jsonl" if d == 200 else "evaluation_tasks_extended.jsonl"
    path = OUT_ROOT / "data" / name
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line]
    tasks = [t for t in tasks if int(t["d"]) == d and int(t["seed"]) == seed]
    if len(tasks) != 50:
        raise ValueError(f"expected 50 tasks for d={d} seed={seed}, got {len(tasks)}")
    return sorted(tasks, key=lambda t: (int(t["episode_idx"]), int(t["start_step"])))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--d", type=int, choices=(100, 150, 200), required=True)
    p.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    kind = "flat_direct_goal" if args.condition == "flat-direct-goal" else "hierarchical"
    checkpoint = H3_RESULTS / f"checkpoints/seed_{args.seed}/{kind}_ema.pt"
    model, meta = h3_model.load_checkpoint(checkpoint, args.device)
    if meta["update"] != 30_000:
        raise ValueError("checkpoint update != 30000")

    tasks = load_tasks(args.d, args.seed)
    normalizers = h3_data.Normalizers.load(H3_RESULTS / "data/normalizers.npz")
    h3_evaluate.H3ClockedBatchPolicy.context = {
        "s2_normalizer": h3_data.S2Normalizer.load(H3_RESULTS / "data/s2_normalizer.npz")
    }
    records, actions = h3_evaluate.run_group(
        model=model,
        condition=args.condition,
        tasks=tasks,
        hdf5=H5,
        normalizers=normalizers,
        checkpoint_sha256=reused.sha256_file(checkpoint),
        input_sha256=reused.sha256_file(OUT_ROOT / "data" / ("evaluation_tasks_d200.jsonl" if args.d == 200 else "evaluation_tasks_extended.jsonl")),
        device=torch.device(args.device),
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shard = OUT_ROOT / f"raw_rollouts_{args.condition}_d{args.d}_m{model.config.seed}_e{args.seed}.jsonl"
    shard.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
    np.savez_compressed(shard.with_suffix(".npz"), **actions)
    event_records = [(record, event) for record in records for event in record["slow_state_events"]]
    events = [x[1] for x in event_records]
    np.savez_compressed(
        shard.with_name(shard.stem + "_instrumentation.npz"),
        condition=np.asarray([args.condition] * len(events)),
        d=np.asarray([args.d] * len(events)),
        model_seed=np.asarray([model.config.seed] * len(events)),
        evaluation_seed=np.asarray([args.seed] * len(events)),
        episode_idx=np.asarray([x[0]["episode_idx"] for x in event_records]),
        start_step=np.asarray([x[0]["start_step"] for x in event_records]),
        boundary=np.asarray([x["boundary"] for x in events]),
        raw_s2=np.asarray([x["raw_s2"] for x in events], np.float32),
        applied_s2=np.asarray([x["applied_s2"] for x in events], np.float32),
        jacobian_frobenius=np.asarray([x["jacobian_frobenius"] for x in events]),
        path_length=np.asarray([x["path_length"] for x in events]),
    )
    n_success = sum(1 for r in records if r["success"])
    print(f"{args.condition} d{args.d} seed{args.seed}: success {n_success}/50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
