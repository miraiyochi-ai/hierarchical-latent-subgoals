#!/usr/bin/env python3
"""Generate H3x evaluation tasks (d in {100,150}) restricted to the test split.

Same sampling rule as H1 stage0 recreate_tasks, except `eligible` is built only
from test-split episodes to avoid training contamination (see README).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
H3_RESULTS = HERE.parents[1] / "results/a2a_h3_state_subgoal_pusht"
OUT_ROOT = HERE.parents[1] / "results/a2a_h3x_horizon_extension"
H5 = Path(os.environ.get("H3X_H5", HERE.parents[1] / "data/pusht/pusht_expert_train.h5"))

EVALUATION = {100: 200, 150: 300}
SEEDS = (42, 43, 44)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    split = json.loads((H3_RESULTS / "data/split_ids.json").read_text())
    test_ids = np.asarray(sorted(split["test_episode_ids"]) if isinstance(split["test_episode_ids"], list) else [], dtype=np.int64)
    with h5py.File(H5, "r", swmr=True) as h:
        lengths_all = np.asarray(h["ep_len"][:], dtype=np.int64)
        offsets_all = np.asarray(h["ep_offset"][:], dtype=np.int64)
        episodes_col = np.asarray(h["episode_idx"][:], dtype=np.int64)
        steps_col = np.asarray(h["step_idx"][:], dtype=np.int64)

    tasks = []
    for d, budget in EVALUATION.items():
        eligible = np.concatenate([
            np.arange(offsets_all[e], offsets_all[e] + lengths_all[e] - d, dtype=np.int64)
            for e in test_ids
            if lengths_all[e] > d
        ])
        print(f"d={d}: eligible rows {len(eligible)} from {sum(1 for e in test_ids if lengths_all[e] > d)} episodes")
        for seed in SEEDS:
            rows = np.sort(eligible[np.random.default_rng(seed).choice(len(eligible), 50, replace=False)])
            group = f"d{d}_seed{seed}"
            for eval_index, row in enumerate(rows.tolist()):
                base = {
                    "group_id": group, "d": d, "seed": seed, "eval_index": eval_index,
                    "global_row": row, "episode_idx": int(episodes_col[row]),
                    "start_step": int(steps_col[row]), "goal_step": int(steps_col[row]) + d,
                    "eval_budget": budget,
                }
                tasks.append({**base, "task_sha256": hashlib.sha256(canonical(base).encode()).hexdigest()})

    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "data/evaluation_tasks_extended.jsonl"
    out.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (OUT_ROOT / "data/evaluation_tasks_extended.sha256").write_text(sha + "\n")
    print(f"{len(tasks)} tasks -> {out}\nsha256 {sha}")


if __name__ == "__main__":
    main()
