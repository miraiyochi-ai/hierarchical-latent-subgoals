"""Fixed 30k-update training; this script never creates paid resources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from common import atomic_json, digest, progress, shared_root, source_hashes
from data import Store
from model import Config, count_parameters
from training import Trainer, save_torch, verify_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--architecture', choices=['flat','hierarchical'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--until', type=int, default=30000)
    a = p.parse_args()
    root = shared_root(a.root)
    if Path(a.name).name != a.name or a.name in ('.','..') or not 1 <= a.until <= 30000:
        raise ValueError('invalid run identity/target')
    run = root / 'runs' / a.name
    run.mkdir(parents=True, exist_ok=False)
    configurations = json.loads((root/'model_configs.json').read_text())
    config = Config(**configurations['configurations'][a.architecture])
    store = Store(root, report=run)
    contract = dict(architecture=a.architecture, config=configurations['configurations'][a.architecture],
                    total_updates=30000, batch_size=128, microbatch_size=32, seed=42,
                    prepared_sha256=digest(root/'prepared.json'), normalization=store.normalization,
                    source_sha256=source_hashes(Path(__file__).parent),
                    lineage_sha256={x.name:digest(x) for x in (Path(__file__).parent/'lineage').glob('*.py')},
                    data_split=dict(train=[0,44], validation=[45,49]), observation_timing='post_action',
                    lower_steps=8, upper_steps=16, learning_rate=1e-4, precision='float32', frozen_features=True)
    trainer = Trainer(config)
    contract['trainable_parameters'] = count_parameters(trainer.model)
    if a.resume:
        trainer.restore(torch.load(a.resume, map_location='cpu', weights_only=False), contract)
    atomic_json(run/'contract.json', contract)
    last_save, started, start_step = time.monotonic(), time.monotonic(), trainer.step
    progress(run, 'train', trainer.step, model=a.architecture)
    with (run/'train.jsonl').open('x') as log:
        while trainer.step < a.until:
            lower = store.sample(trainer.sampler, 128)
            upper = store.sample(trainer.sampler, 128, upper_only=True)
            metrics = trainer.update(lower, upper)
            metrics['time'] = time.time()
            log.write(json.dumps(metrics, allow_nan=False)+'\n'); log.flush()
            if trainer.step <= 2 or trainer.step % 10 == 0:
                speed = (trainer.step-start_step)/(time.monotonic()-started)
                progress(run, 'train', trainer.step, metrics=metrics, updates_per_second=speed,
                         remaining_seconds=(a.until-trainer.step)/speed)
            if trainer.step <= 2 or trainer.step % 500 == 0 or time.monotonic()-last_save >= 300 or trainer.step == a.until:
                checkpoint = run/'latest.pt'
                save_torch(checkpoint, trainer.state(contract))
                verified = verify_state(checkpoint, trainer.step, contract)
                atomic_json(run/'checkpoint.json', dict(**verified, sha256=digest(checkpoint), time=time.time()))
                last_save = time.monotonic()
            if trainer.step in (5000,15000,30000) or trainer.step == a.until:
                destination = run/f'ema_{trainer.step:06d}.pt'
                if destination.exists():
                    raise FileExistsError('immutable evaluation snapshot exists')
                save_torch(destination, dict(checkpoint_kind='evaluation_only', weights_kind='ema',
                    model=trainer.ema, config=contract['config'], contract=contract, step=trainer.step))
                atomic_json(destination.with_suffix('.json'), dict(sha256=digest(destination), step=trainer.step))
    final = verify_state(run/'latest.pt', a.until, contract)
    atomic_json(run/'result.json', dict(**final, state='training_complete' if a.until==30000 else 'training_segment_complete',
        checkpoint_sha256=digest(run/'latest.pt'), elapsed_seconds=time.monotonic()-started,
        closed_loop_evaluation_complete=False))
    progress(run, 'training_complete' if a.until==30000 else 'training_segment_complete', trainer.step)


if __name__ == '__main__':
    main()
