"""Fixed-budget H3 optimization and complete, auditable resume state."""
from __future__ import annotations
import copy
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from model import Policy, Config


def configure(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def multiplier(step, total):
    if step < 500:
        return (step + 1) / 500
    return .5 * (1 + math.cos(math.pi * min(1., (step - 500 + 1) / (total - 500))))


class Trainer:
    def __init__(self, config, device='cuda', total=30000):
        configure(config.seed)
        self.device, self.total, self.step = torch.device(device), total, 0
        self.model = Policy(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4,
                                           betas=(.95, .999), eps=1e-8, weight_decay=1e-6)
        self.ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.lower_rng = torch.Generator(device=device).manual_seed(config.seed+100)
        self.upper_rng = torch.Generator(device=device).manual_seed(config.seed+200)
        self.sampler = np.random.default_rng(config.seed+300)
        self.draws = 0

    def update(self, lower, upper, microbatch=32):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = len(lower['language'])
        if batch % microbatch:
            raise ValueError('microbatch must divide the effective batch')
        for group in self.optimizer.param_groups:
            group['lr'] = 1e-4 * multiplier(self.step, self.total)
        metrics = {}
        for start in range(0, batch, microbatch):
            lo = {k: v[start:start+microbatch].to(self.device) for k, v in lower.items()}
            up = {k: v[start:start+microbatch].to(self.device) for k, v in upper.items()}
            condition = self.model.predict_subgoal(lo) if self.model.layer2 is not None else lo['language']
            loss1 = self.model.lower_loss(lo, condition, self.lower_rng)
            loss2 = self.model.upper_loss(up, self.upper_rng) if self.model.layer2 is not None else None
            objective = loss1['total'] + (loss2['total'] if loss2 is not None else 0)
            if not torch.isfinite(objective):
                raise FloatingPointError('nonfinite objective; no update performed')
            (objective * microbatch / batch).backward()
            values = {'total': objective, 'lower': loss1['total'], 'action_l1': loss1['flow_action_l1']}
            if loss2 is not None:
                values.update(upper=loss2['total'], upper_endpoint=loss2['endpoint'])
            for key, val in values.items():
                metrics[key] = metrics.get(key, 0.) + float(val.detach()) * microbatch / batch
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1; self.draws += batch
        decay = min(.9999, 1 - (self.step + 1)**-.75)
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                if value.is_floating_point():
                    self.ema[key].lerp_(value.detach(), 1-decay)
                else:
                    self.ema[key].copy_(value)
        metrics.update(step=self.step, grad_norm=float(norm), learning_rate=self.optimizer.param_groups[0]['lr'])
        return metrics

    def state(self, contract):
        return dict(checkpoint_kind='full_training_state', model=self.model.state_dict(), ema=self.ema,
                    optimizer=self.optimizer.state_dict(), config=asdict(self.model.config), contract=contract,
                    step=self.step, total=self.total, draws=self.draws,
                    scheduler=dict(rule='warmup500_cosine', step=self.step, total=self.total),
                    lower_rng=self.lower_rng.get_state(), upper_rng=self.upper_rng.get_state(),
                    sampler=copy.deepcopy(self.sampler.bit_generator.state),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())

    def restore(self, saved, contract):
        if saved['checkpoint_kind'] != 'full_training_state' or saved['contract'] != contract:
            raise ValueError('incompatible checkpoint contract')
        if saved['config'] != asdict(self.model.config) or saved['total'] != self.total:
            raise ValueError('configuration/schedule mismatch')
        if saved['scheduler'] != dict(rule='warmup500_cosine', step=saved['step'], total=self.total):
            raise ValueError('scheduler mismatch')
        self.model.load_state_dict(saved['model']); self.optimizer.load_state_dict(saved['optimizer'])
        self.ema = {k: v.to(self.device).clone() for k, v in saved['ema'].items()}
        self.step, self.draws = saved['step'], saved['draws']
        self.lower_rng.set_state(saved['lower_rng'].cpu()); self.upper_rng.set_state(saved['upper_rng'].cpu())
        self.sampler.bit_generator.state = saved['sampler']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])


def save_torch(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        torch.save(data, stream)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def verify_state(path, expected_step, expected_contract):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    required = {'model', 'ema', 'optimizer', 'sampler', 'lower_rng', 'upper_rng', 'torch_rng', 'cuda_rng',
                'numpy_rng', 'python_rng', 'scheduler', 'step', 'draws'}
    if not required <= saved.keys() or saved['step'] != expected_step or saved['contract'] != expected_contract:
        raise ValueError('incomplete/mismatched training snapshot')
    for group in ('model', 'ema'):
        if any(not torch.isfinite(v).all() for v in saved[group].values()):
            raise FloatingPointError('nonfinite snapshot')
    if expected_step and not saved['optimizer']['state']:
        raise ValueError('empty optimizer state')
    return dict(step=saved['step'], draws=saved['draws'], full_state_verified=True)
