"""Read-only paired comparison: old short-horizon vs new extended-horizon training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HORIZONS = (25, 50, 75, 100, 150)


def read_cell(root, condition, horizon):
    folder = 'evaluation_shards' if horizon <= 75 else 'h3x'
    path = Path(root) / folder / f'raw_rollouts_{condition}_d{horizon}_m42_e42.jsonl'
    raw = path.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError(f'incomplete output: {path}')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 50:
        raise ValueError(f'exactly 50 episodes required: {path}')
    keyed = {}
    checkpoints, inputs = set(), set()
    for row in rows:
        for key, value in [('condition', condition), ('d', horizon), ('model_seed', 42), ('evaluation_seed', 42)]:
            if row[key] != value:
                raise ValueError(f'{key} mismatch: {path}')
        if type(row['success']) is not bool:
            raise ValueError('invalid success value')
        key = (row['episode_idx'], row['start_step'], row['task_sha256'])
        if key in keyed:
            raise ValueError('duplicate episode')
        keyed[key] = row['success']
        checkpoints.add(row['checkpoint_sha256'])
        inputs.add(row['input_sha256'])
    if len(checkpoints) != 1 or len(inputs) != 1:
        raise ValueError('mixed weights or inputs within a cell')
    return keyed, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                      checkpoint_sha256=checkpoints.pop(), input_sha256=inputs.pop())


def paired(hierarchical, flat):
    if not hierarchical or hierarchical.keys() != flat.keys():
        raise ValueError('the same tasks must be evaluated in both conditions')
    n = len(hierarchical)
    h, f = sum(hierarchical.values()), sum(flat.values())
    h_only = sum(hierarchical[k] and not flat[k] for k in hierarchical)
    f_only = sum(flat[k] and not hierarchical[k] for k in hierarchical)
    discordant = h_only + f_only
    exact_p = (min(1., 2 * sum(math.comb(discordant, i)
                             for i in range(min(h_only, f_only) + 1)) / 2**discordant)
               if discordant else 1.)
    return dict(episodes=n, hierarchical_successes=h, flat_successes=f,
                hierarchical_success_rate=h/n, flat_success_rate=f/n,
                hierarchical_minus_flat_points=100*(h-f)/n,
                hierarchical_only=h_only, flat_only=f_only,
                exact_paired_p_two_sided_unadjusted=exact_p,
                descriptive_winner='hierarchical' if h > f else 'flat' if f > h else 'tie')


def compare(roots):
    cells, provenance = {}, {}
    for architecture in ('hierarchical', 'flat'):
        old, new = Path(roots['old_'+architecture]), Path(roots['new_'+architecture])
        condition = 'hierarchical' if architecture == 'hierarchical' else 'flat-direct-goal'
        manifest = json.loads((new/'train150_manifest.json').read_text())
        if any(manifest.get(key) != value for key, value in
               [('condition', condition), ('seed', 42), ('updates', 30000),
                ('training_horizons', list(HORIZONS)), ('initialization', 'from_scratch_not_finetuning')]):
            raise ValueError('new training manifest does not match the registered experiment')
        protocol = json.loads((old/'protocol.json').read_text())
        if protocol.get('horizons') != [25, 50, 75] or protocol.get('updates') != 30000:
            raise ValueError('old training reference is not the three-horizon experiment')
        for filename in ('parameter_manifest.json', 'source_manifest.json',
                         f'effective_configs/{condition.replace("-", "_")}_seed_42.json'):
            if json.loads((old/filename).read_text()) != json.loads((new/filename).read_text()):
                raise ValueError(f'architecture, source or training settings changed: {filename}')
    for horizon in HORIZONS:
        values, evidence = {}, {}
        for label, root in roots.items():
            condition = 'predicted' if label.endswith('hierarchical') else 'flat-direct-goal'
            values[label], evidence[label] = read_cell(root, condition, horizon)
        first = next(iter(values.values()))
        if any(value.keys() != first.keys() for value in values.values()):
            raise ValueError('old/new and hierarchical/flat evaluation tasks differ')
        if len({v['input_sha256'] for v in evidence.values()}) != 1:
            raise ValueError('old/new input task files differ')
        cell = {}
        for training in ('old', 'new'):
            cell[training] = paired(values[training+'_hierarchical'], values[training+'_flat'])
        cell['change_in_hierarchical_minus_flat_points'] = (
            cell['new']['hierarchical_minus_flat_points'] - cell['old']['hierarchical_minus_flat_points'])
        cells[str(horizon)] = cell
        provenance[str(horizon)] = evidence
    return dict(state='comparison_complete', seed=42, cells=cells, provenance=provenance,
                primary_horizons=[100, 150], supplemental_horizons=[25, 50, 75],
                claim_boundary='One training seed; paired episode p-values do not measure training-seed uncertainty. '
                'Raw p-values are exploratory, not a multiple-test-adjusted superiority claim. '
                'Fixed total updates: extending the balanced mixture also changes per-horizon exposure.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-hierarchical', 'old-flat', 'new-hierarchical', 'new-flat'):
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(compare(vars(parser.parse_args())), indent=2, allow_nan=False))
