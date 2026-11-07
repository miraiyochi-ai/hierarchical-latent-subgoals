#!/usr/bin/env python3
"""H3 joint/flat training with reference and uint8 optimized supply paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

import h3_data
import h3_model

H1_DIR=Path(__file__).resolve().parents[1]/"a2a_h1_clockwork_pusht_gate"
if str(H1_DIR) not in sys.path: sys.path.insert(0,str(H1_DIR))
import h1_train as ring  # noqa: E402

UPDATES = 30_000
WARMUP_UPDATES = 500
LEARNING_RATE = 1e-4
MICROBATCH_SIZE = 32
GRADIENT_CLIP = 1.0
LAYER2_WEIGHT = 1.0


@dataclass(frozen=True)
class RunConfig:
    updates: int
    batch_size: int
    microbatch_size: int
    device: str


def effective_data_workers(profile: str, requested: int | None) -> int:
    if requested is not None and requested < 0: raise ValueError("data workers cannot be negative")
    if requested is not None: return requested
    return 0 if profile == "mac_preflight" else min(8, max(1, (os.cpu_count() or 2) // 2))


def configure_determinism() -> None:
    # CUBLAS_WORKSPACE_CONFIGだけでは無効。G1のref/ring等価比較は同一計算の
    # ビット再現を前提とするため、決定論的カーネルを強制する。
    # flash/mem-efficient SDPAのbackwardはuse_deterministic_algorithmsの対象外で
    # 非決定的なため、mathバックエンドへ固定する(系列長<=16で速度影響は無視できる)。
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def configure_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    configure_determinism()


def learning_rate_multiplier(update: int, total: int = UPDATES) -> float:
    if update < WARMUP_UPDATES: return (update + 1) / WARMUP_UPDATES
    progress = (update - WARMUP_UPDATES + 1) / (total - WARMUP_UPDATES)
    return .5 * (1 + math.cos(math.pi * min(1.0, progress)))


class PowerEMA:
    def __init__(self, model: torch.nn.Module) -> None:
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def update(self, model: torch.nn.Module, update: int) -> float:
        decay = min(0.9999, 1 - (update + 1) ** -0.75)
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if self.state[name].is_floating_point(): self.state[name].lerp_(value.detach(), 1-decay)
                else: self.state[name].copy_(value)
        return decay


class ReferenceBatchBuilder:
    def __init__(self, reader: h3_data.H3HDF5Reader, l1, l2, seed: int, batch_size: int) -> None:
        self.reader = reader; self.l1 = h3_data.CyclicSpecSampler(l1, seed); self.l2 = h3_data.BoundarySampler(l2, seed); self.batch_size = batch_size

    def update(self, index: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        l1_specs = []
        for d, count in h3_data.balanced_horizon_counts(index, self.batch_size).items(): l1_specs.extend(self.l1.take(d, count))
        return h3_data.collate([self.reader.layer1(x) for x in l1_specs]), h3_data.collate([self.reader.layer2(x) for x in self.l2.take_update(index, self.batch_size)])

    def close(self) -> None: self.reader.close()


class Uint8RingBatchBuilder(ReferenceBatchBuilder):
    """Pinned host-ring facade; slots are released only after their CUDA event."""
    def __init__(self, *args, slots: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs); self.slots = slots; self.events: list[torch.cuda.Event | None] = [None] * slots; self.sequence = 0

    def update(self, index: int):
        slot = self.sequence % self.slots; event = self.events[slot]
        if event is not None: event.synchronize()
        batches = super().update(index)
        if torch.cuda.is_available():
            batches = tuple({name: value.pin_memory() for name, value in batch.items()} for batch in batches)
        self.sequence += 1
        return batches

    def mark_consumed(self) -> None:
        if torch.cuda.is_available():
            slot = (self.sequence - 1) % self.slots; self.events[slot] = torch.cuda.Event(); self.events[slot].record()

def _h3_batch_schema(layer:int,batch_size:int,image_size:int):
    image=lambda steps:(batch_size,steps,3,image_size,image_size)
    common={"episode_idx":((batch_size,),torch.int64),"task_start_step":((batch_size,),torch.int64),"d":((batch_size,),torch.int64),"boundary":((batch_size,),torch.int64)}
    single=(batch_size,3,image_size,image_size); l2={"observation_window":(image(16),torch.uint8),"current_image":(single,torch.uint8),"goal_image":(single,torch.uint8),"action_history":((batch_size,16,2),torch.float32),"history_mask":((batch_size,1),torch.float32),"remaining_horizon":((batch_size,1),torch.float32)}
    if layer==1:return {"head_cam":(image(8),torch.uint8),"agent_pos":((batch_size,8,2),torch.float32),"future_actions":((batch_size,8,2),torch.float32),"u":((batch_size,),torch.int64),"s2_id_hash":((batch_size,),torch.int64),**common,**l2}
    if layer==2:return {**l2,"teacher_image":(single,torch.uint8),**common,"teacher_step":((batch_size,),torch.int64)}
    raise ValueError("batch layer must be one or two")
def _h3_map(spec):
    dtype={"float32":torch.float32,"int64":torch.int64,"uint8":torch.uint8}[spec.dtype]; return torch.from_file(spec.path,shared=True,size=math.prod(spec.shape),dtype=dtype).reshape(spec.shape)
def _h3_create(root,*,layer,slot,batch_size,image_size):
    tensors={}; specs={}
    for name,(shape,dtype) in _h3_batch_schema(layer,batch_size,image_size).items():
        path=root/f"layer{layer}_slot{slot}_{name}.bin"; path.touch(mode=0o600); spec=ring._MappedTensorSpec(str(path),shape,str(dtype).split(".")[-1]); tensors[name]=_h3_map(spec); specs[name]=spec
    return tensors,specs
def _h3_worker(connection,hdf5,lengths,offsets,normalizers,image_size,layer1_specs,layer2_specs):
    torch.set_num_threads(1); table=h3_data.EpisodeTable(np.asarray(lengths),np.asarray(offsets)); reader=h3_data.H3HDF5Reader(Path(hdf5),table,normalizers); slots1=tuple({n:_h3_map(s) for n,s in specs.items()} for specs in layer1_specs); slots2=tuple({n:_h3_map(s) for n,s in specs.items()} for specs in layer2_specs)
    try:
        reader.open()
        while True:
            message=connection.recv()
            if message is None:break
            task_id,layer,slot,row_start,specs=message
            try:
                destination=slots1[slot] if layer==1 else slots2[slot]
                for offset,spec in enumerate(specs):
                    record=reader.layer1(spec) if layer==1 else reader.layer2(spec)
                    for name,value in record.items(): destination[name][row_start+offset].copy_(value)
                connection.send((task_id,None))
            except BaseException: connection.send((task_id,traceback.format_exc()))
    finally:reader.close();connection.close()
class SharedRingBatchBuilder(ring.OrderedParallelBatchBuilder):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); self.cuda_event=None
    def _submit(self,update:int)->None:
        slot=update%self.prefetch_depth
        if self.active_slot==slot or any(x.slot==slot for x in self.pending):raise RuntimeError("mapped batch slot was reused before consumption")
        l1=[]
        for d,count in h3_data.balanced_horizon_counts(update,self.batch_size).items():l1.extend(self.layer1_sampler.take(d,count))
        first=self._submit_layer(1,slot,l1); second=self._submit_layer(2,slot,self.layer2_sampler.take_update(update,self.batch_size)); self.pending.append(ring._PendingUpdate(update,slot,first,second))
    def update(self,index:int):
        if self.cuda_event is not None:self.cuda_event.synchronize();self.cuda_event=None
        one,two=self.get(index); assert two is not None
        if torch.cuda.is_available(): return ({k:v.pin_memory() for k,v in one.items()},{k:v.pin_memory() for k,v in two.items()})
        return one,two
    def mark_consumed(self):
        if torch.cuda.is_available():self.cuda_event=torch.cuda.Event();self.cuda_event.record()
    def close(self):
        if getattr(self,"cuda_event",None) is not None:self.cuda_event.synchronize();self.cuda_event=None
        super().close()

ring._batch_schema=_h3_batch_schema; ring._map_tensor=_h3_map; ring._create_mapped_batch=_h3_create; ring._batch_worker_main=_h3_worker


def move(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    # .contiguous(): 参照経路の転置ビュー(非連続stride)とリング経路の標準連続で
    # カーネルの計算順が変わりULP差が出るため、消費点でレイアウトを正規化する。
    return {name: value.contiguous().to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}


def _slices(batch: Mapping[str, Tensor], size: int) -> Iterable[dict[str, Tensor]]:
    count = next(iter(batch.values())).shape[0]
    for start in range(0, count, size): yield {name: value[start:start+size] for name, value in batch.items()}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(value, sort_keys=True) + "\n")
def _write_identical_once(path:Path,value:Any)->None:
 text=json.dumps(value,sort_keys=True,indent=2)
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_text(encoding="utf-8")!=text: raise RuntimeError(f"h3_training_blocked: conflicting immutable artifact {path}")
 else: path.write_text(text,encoding="utf-8")


def train_one(model: h3_model.H3Policy, builder: ReferenceBatchBuilder, normalizer: h3_data.S2Normalizer, config: RunConfig, loss_path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    configure_determinism()
    device = torch.device(config.device); model.to(device); model.train(); model.lewm.eval()
    before_sha = h3_model.module_state_sha256(model.lewm)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, betas=(.95,.999), eps=1e-8, weight_decay=1e-6)
    ema = PowerEMA(model); generator = torch.Generator(device=device).manual_seed(model.config.seed)
    gradient_seen={name:False for name,p in model.named_parameters() if p.requires_grad}
    mu = torch.as_tensor(normalizer.mu, dtype=torch.float32, device=device); sigma = torch.as_tensor(normalizer.sigma, dtype=torch.float32, device=device)
    for update in range(config.updates):
        l1_cpu, l2_cpu = builder.update(update); optimizer.zero_grad(set_to_none=True); sums: dict[str, float] = {}
        l1_parts, l2_parts = list(_slices(l1_cpu, config.microbatch_size)), list(_slices(l2_cpu, config.microbatch_size))
        for l1_raw, l2_raw in zip(l1_parts, l2_parts):
            l1, l2 = move(l1_raw, device), move(l2_raw, device)
            if model.config.hierarchical:
                slow = model.predict_s2(l1, mu, sigma)  # L1 gradients continue into layer two.
                first = model.layer1_loss(l1, slow, generator); second = model.layer2_loss(l2, mu, sigma, generator)
                total = first["total"] + LAYER2_WEIGHT * second["total"]
                values = {**{f"l1_{k}": v for k,v in first.items()}, **{f"l2_{k}": v for k,v in second.items() if k not in {"prediction", "path_length"}}, "total": total}
            else:
                goal = model.encode_s2(l1["goal_image"], mu, sigma)
                first = model.layer1_loss(l1, goal, generator); total = first["total"]
                values = {**{f"l1_{k}": v for k,v in first.items()}, "total": total}
            (total / len(l1_parts)).backward()
            for name, value in values.items(): sums[name] = sums.get(name, 0.0) + float(value.detach()) / len(l1_parts)
        for name,p in model.named_parameters():
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0: gradient_seen[name]=True
        grad = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), GRADIENT_CLIP)
        if not torch.isfinite(grad): raise RuntimeError("h3_training_blocked: non-finite gradient")
        factor = learning_rate_multiplier(update, config.updates); [group.update(lr=LEARNING_RATE*factor) for group in optimizer.param_groups]
        optimizer.step(); decay = ema.update(model, update)
        if hasattr(builder,"mark_consumed"): builder.mark_consumed()
        _append_jsonl(loss_path, {"update": update+1, "condition": "hierarchical" if model.config.hierarchical else "flat-direct-goal", "model_seed":model.config.seed, "losses": sums, "gradient_norm": float(grad), "lr": LEARNING_RATE*factor, "ema_decay": decay})
    after_sha = h3_model.module_state_sha256(model.lewm)
    if before_sha != after_sha: raise RuntimeError("h3_training_blocked: frozen LeWM changed")
    return ema.state, {"updates": config.updates, "lewm_sha_before": before_sha, "lewm_sha_after": after_sha,"all_trainable_parameters_received_nonzero_gradient":all(gradient_seen.values()),"missing_gradient_parameters":[name for name,seen in gradient_seen.items() if not seen],"frozen_lewm_gradient_count":sum(p.grad is not None for p in model.lewm.parameters())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--profile", choices=("mac_preflight","gpu_full"), required=True); parser.add_argument("--condition", choices=("hierarchical","flat-direct-goal"), required=True); parser.add_argument("--seed", type=int, choices=h3_model.MODEL_SEEDS, default=42); parser.add_argument("--hdf5", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--device"); parser.add_argument("--supply", choices=("reference","optimized"), default="optimized"); parser.add_argument("--data-workers", type=int); return parser.parse_args()


def main() -> int:
    args = parse_args(); device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    config = RunConfig(8 if args.profile == "mac_preflight" else UPDATES, 8 if args.profile == "mac_preflight" else 128, 2 if args.profile == "mac_preflight" else MICROBATCH_SIZE, device)
    splits = h3_data.load_split_ids(args.data_root / "split_ids.json"); table = h3_data.EpisodeTable.from_hdf5(args.hdf5); normalizers = h3_data.Normalizers.load(args.data_root / "normalizers.npz"); s2norm = h3_data.S2Normalizer.load(args.data_root / "s2_normalizer.npz")
    ids = splits["train"][:64] if args.profile == "mac_preflight" else splits["train"]
    l1, l2 = h3_data.enumerate_layer1_specs(table, ids), h3_data.enumerate_layer2_specs(table, ids)
    workers=effective_data_workers(args.profile,args.data_workers)
    if args.supply=="optimized" and workers:
        builder=SharedRingBatchBuilder(hdf5=args.hdf5,table=table,normalizers=normalizers,image_size=224,layer1_sampler=h3_data.CyclicSpecSampler(l1,args.seed),layer2_sampler=h3_data.BoundarySampler(l2,args.seed),batch_size=config.batch_size,total_updates=config.updates,workers=workers)
    else: builder=ReferenceBatchBuilder(h3_data.H3HDF5Reader(args.hdf5,table,normalizers),l1,l2,args.seed,config.batch_size)
    hierarchical, flat, manifest = h3_model.build_capacity_matched_pair(args.seed); model = hierarchical if args.condition == "hierarchical" else flat
    args.output_root.mkdir(parents=True, exist_ok=True); _write_identical_once(args.output_root/"parameter_manifest.json",manifest)
    here=Path(__file__).resolve().parent; sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}; _write_identical_once(args.output_root/"source_manifest.json",{"experiment":h3_model.EXPERIMENT_ID,"sources":sources,"readme_sha256":hashlib.sha256((here/"README.md").read_bytes()).hexdigest()}); _write_identical_once(args.output_root/"protocol.json",{"clock":16,"prediction_horizon":16,"hold":16,"layer1_chunk":8,"euler_steps":6,"updates":30_000,"batch":{"layer1":128,"layer2":128},"model_seeds":[42,43,44],"horizons":[25,50,75],"conditions":list(("predicted","oracle","identity","goal-swap-recompute","zero","flat-direct-goal"))})
    effective={"condition":args.condition,"seed":args.seed,"run":asdict(config),"model":asdict(model.config),"supply":args.supply,"optimizer":{"name":"AdamW","lr":LEARNING_RATE,"betas":[.95,.999],"eps":1e-8,"weight_decay":1e-6},"warmup":WARMUP_UPDATES,"ema":{"inverse_gamma":1,"power":.75,"max":.9999}}; config_path=args.output_root/(f"g3_mac_preflight/effective_{args.condition}.json" if args.profile=="mac_preflight" else f"effective_configs/{args.condition.replace('-','_')}_seed_{args.seed}.json"); _write_identical_once(config_path,effective)
    loss = args.output_root / ("g3_mac_preflight/raw_losses.jsonl" if args.profile == "mac_preflight" else "training_losses.jsonl")
    try: ema, summary = train_one(model, builder, s2norm, config, loss)
    finally: builder.close()
    checkpoint = args.output_root / (f"g3_mac_preflight/checkpoints/{args.condition}_ema.pt" if args.profile == "mac_preflight" else f"checkpoints/seed_{args.seed}/{args.condition.replace('-', '_')}_ema.pt")
    h3_model.save_checkpoint(checkpoint, model, config.updates, ema); checkpoint.with_suffix(".summary.json").write_text(json.dumps({**summary, "run": asdict(config)}, indent=2), encoding="utf-8"); return 0


if __name__ == "__main__": raise SystemExit(main())
