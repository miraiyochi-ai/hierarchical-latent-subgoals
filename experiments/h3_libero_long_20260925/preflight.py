"""Real-data GPU gate: capacity, gradients, finite update and exact resume."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import atomic_json, digest, progress, shared_root, source_hashes
from data import Store
from model import Config, matched_configs, count_parameters
from training import Trainer, configure, save_torch, verify_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name',default='preflight_v1')
    a = p.parse_args()
    root = shared_root(a.root)
    if Path(a.name).name!=a.name: raise ValueError('invalid gate identity')
    output = root/a.name
    output.mkdir(exist_ok=False)
    configure(42)
    progress(output, 'configure', 0)
    configs, counts = matched_configs()
    atomic_json(root/'model_configs.json', dict(configurations=configs, counts=counts))
    store = Store(root, report=output)
    results = []
    for architecture in ('flat', 'hierarchical'):
        trainer = Trainer(Config(**configs[architecture]))
        if count_parameters(trainer.model) != counts[architecture]:
            raise ValueError('analytic and actual capacity disagree')
        lower = store.sample(np.random.default_rng(100), 4)
        upper = store.sample(np.random.default_rng(200), 4, upper_only=True)
        # H3 modulation is zero-initialized; connectivity is measurable after
        # one real update, not at the untouched initialization.
        trainer.update(lower, upper, microbatch=2)
        gradient = None
        if architecture == 'hierarchical':
            batch = {k:v.cuda() for k,v in lower.items()}
            trainer.model.zero_grad(set_to_none=True)
            predicted = trainer.model.predict_subgoal(batch)
            loss = trainer.model.lower_loss(batch, predicted, torch.Generator(device='cuda').manual_seed(77))['total']
            loss.backward()
            gradient = sum(float(x.grad.detach().norm()) for x in trainer.model.layer2.parameters() if x.grad is not None)
            if not np.isfinite(gradient) or gradient <= 0:
                raise ValueError('lower loss does not train the upper generator')
            trainer.model.zero_grad(set_to_none=True)
        contract = dict(preflight=True, architecture=architecture)
        checkpoint = output/f'{architecture}_step1.pt'
        save_torch(checkpoint, trainer.state(contract))
        verify_state(checkpoint, 1, contract)
        next_lower=store.sample(trainer.sampler,4)
        next_upper=store.sample(trainer.sampler,4,upper_only=True)
        reference = trainer.update(next_lower, next_upper, microbatch=2)
        expected = {k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
        expected_ema = {k:v.detach().cpu().clone() for k,v in trainer.ema.items()}
        del trainer
        gc.collect(); torch.cuda.empty_cache()
        resumed = Trainer(Config(**configs[architecture]))
        resumed.restore(torch.load(checkpoint, map_location='cpu', weights_only=False), contract)
        resumed_lower=store.sample(resumed.sampler,4)
        resumed_upper=store.sample(resumed.sampler,4,upper_only=True)
        if any(not torch.equal(next_lower[k],resumed_lower[k]) or not torch.equal(next_upper[k],resumed_upper[k]) for k in next_lower):
            raise ValueError('sampler resume mismatch')
        observed = resumed.update(resumed_lower, resumed_upper, microbatch=2)
        if reference != observed or any(not torch.equal(v.cpu(),expected[k]) for k,v in resumed.model.state_dict().items()) or any(not torch.equal(v.cpu(),expected_ema[k]) for k,v in resumed.ema.items()):
            raise ValueError('two-update save/resume is not bit-identical')
        # Time intended full effective batch after the acceptance check.
        lo = store.sample(np.random.default_rng(300),128)
        up = store.sample(np.random.default_rng(400),128,upper_only=True)
        torch.cuda.synchronize(); started=time.monotonic()
        measurements=[]
        for _ in range(3):
            measurements.append(resumed.update(lo,up))
        torch.cuda.synchronize()
        seconds=(time.monotonic()-started)/3
        results.append(dict(architecture=architecture, parameters=counts[architecture],
                            exact_resume=True, lower_to_upper_gradient_norm=gradient,
                            seconds_per_update=seconds, expected_30000_update_seconds=seconds*30000,
                            final_loss=measurements[-1]['total']))
        atomic_json(output/'partial.json',dict(results=results))
        progress(output,'verified_model',len(results),result=results[-1])
        del resumed,expected,expected_ema
        gc.collect(); torch.cuda.empty_cache()
    atomic_json(output/'result.json',dict(state='gpu_training_gate_passed',results=results,
                source_note='synthetic resume uses real cached data; no scientific success measured',
                prepared_sha256=digest(root/'prepared.json'),source_sha256=source_hashes(Path(__file__).parent), simulator_gate_passed=False))
    progress(output,'complete',2)


if __name__=='__main__':
    main()
