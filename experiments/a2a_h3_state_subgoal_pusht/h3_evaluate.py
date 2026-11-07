#!/usr/bin/env python3
"""H3 six-condition closed-loop evaluation with a 16/16 task-relative clock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.nn import functional as F

import h3_data
import h3_model

H1_DIR = Path(__file__).resolve().parents[1] / "a2a_h1_clockwork_pusht_gate"
if str(H1_DIR) not in sys.path: sys.path.insert(0, str(H1_DIR))
import h1_evaluate as reused  # noqa: E402

CONDITIONS = ("predicted", "oracle", "identity", "goal-swap-recompute", "zero", "flat-direct-goal")
BUDGETS = {25: 50, 50: 100, 75: 150}


def fixed_cyclic_partners(tasks: Sequence[Mapping[str, Any]]) -> list[int]:
    order = sorted(range(len(tasks)), key=lambda i: (str(tasks[i].get("task_sha256", "")), int(tasks[i]["episode_idx"]), int(tasks[i]["start_step"])))
    if len(order) < 2: raise ValueError("goal swap requires at least two tasks")
    partners = [-1] * len(tasks)
    for position, index in enumerate(order): partners[index] = order[(position + 1) % len(order)]
    if any(index == partner for index, partner in enumerate(partners)): raise RuntimeError("goal swap self-match")
    return partners


def should_update_s2(boundary: int, d: int) -> bool:
    return boundary % 16 == 0 and boundary < d


def held_boundary(u: int, d: int) -> int:
    return min(16 * (u // 16), 16 * ((d - 1) // 16))


class H3ClockedBatchPolicy:
    context: dict[str, Any] = {}
    last_instance: "H3ClockedBatchPolicy | None" = None

    def __init__(self, model, condition, goals, initial_images, initial_agent, normalizers, tasks, device, nearest=None):
        if condition not in CONDITIONS or nearest is not None: raise ValueError("unregistered H3 condition")
        if (condition == "flat-direct-goal") == model.config.hierarchical: raise ValueError("condition/checkpoint architecture mismatch")
        self.model = model.to(device).eval(); self.condition = condition; self.tasks = tasks; self.normalizers = normalizers; self.device = device
        donor = self.context.get("swap_donor_goal")
        if condition != "goal-swap-recompute": self.partners = [-1] * len(tasks)
        elif donor is not None and len(tasks) == 1:
            # mac-preflight: 交換相手のgoalは定義上2課題目が必要だが、登録の18話(各d先頭1件)を
            # 保つため、同dの2件目をgoal提供者としてだけ読み込み、実行課題には加えない。
            goals = np.concatenate([np.asarray(goals), donor[None]], axis=0); self.partners = [1]
        else: self.partners = fixed_cyclic_partners(tasks)
        self.goals = goals; self.image_history = [[x.copy()] * 16 for x in initial_images]; self.agent_history = [[x.copy()] * 8 for x in initial_agent]
        self.action_history: list[list[np.ndarray]] = [[] for _ in tasks]; self.queues: list[list[np.ndarray]] = [[] for _ in tasks]; self.steps = np.zeros(len(tasks), np.int64)
        self.s2 = np.zeros((len(tasks), h3_model.S2_DIM), np.float32); self.raw_s2 = self.s2.copy(); self.slow_events: list[list[dict[str, Any]]] = [[] for _ in tasks]; self.chunk_events: list[list[dict[str,Any]]] = [[] for _ in tasks]
        self.mu = torch.as_tensor(self.context["s2_normalizer"].mu, dtype=torch.float32, device=device); self.sigma = torch.as_tensor(self.context["s2_normalizer"].sigma, dtype=torch.float32, device=device)
        self.nearest_refs: list[list[tuple[int, int]]] = [[] for _ in tasks]
        type(self).last_instance = self

    @property
    def clocks(self):
        # h1のEpisodeClock.updates(s²更新回数)と同義のビュー。h1側run_groupの記録組み立てが参照する。
        import types
        return [types.SimpleNamespace(updates=len(events)) for events in self.slow_events]

    def _head_images(self, values: np.ndarray) -> torch.Tensor:
        value = torch.from_numpy(values).to(self.device)
        if value.dtype == torch.uint8: value = value.float().div(255)
        flat = value.reshape(-1, *value.shape[-3:])
        if flat.shape[-2:] != (224,224): flat = F.interpolate(flat, (224,224), mode="bilinear", align_corners=False)
        return flat.reshape(*value.shape[:-3], 3, 224, 224)

    def _l2_batch(self, indices: Sequence[int], goals: np.ndarray, boundary: int) -> dict[str, torch.Tensor]:
        windows = np.stack([np.stack(self.image_history[i][-16:]) for i in indices])
        actions = np.zeros((len(indices),16,2), np.float32); masks = np.zeros((len(indices),1), np.float32)
        if boundary > 0:
            for local, index in enumerate(indices): actions[local] = h3_data.normalize_action(np.stack(self.action_history[index][-16:])); masks[local] = 1
        return {"observation_window": torch.from_numpy(windows).to(self.device), "current_image": torch.from_numpy(np.stack([self.image_history[i][-1] for i in indices])).to(self.device), "goal_image": torch.from_numpy(goals).to(self.device), "action_history": torch.from_numpy(actions).to(self.device), "history_mask": torch.from_numpy(masks).to(self.device), "remaining_horizon": torch.tensor([[min(16, int(self.tasks[i]["d"])-boundary)/16] for i in indices], device=self.device)}

    def _oracle(self, indices: Sequence[int], boundary: int) -> torch.Tensor:
        images = []
        for index in indices:
            task = self.tasks[index]; key = (int(task["episode_idx"]), min(int(task["start_step"])+boundary+16, int(task["goal_step"])))
            images.append(self.context["oracle_images"][key])
        return (self.model.lewm(torch.from_numpy(np.stack(images)).to(self.device)) - self.mu) / self.sigma

    def _jacobian_norms(self, indices: Sequence[int], applied: torch.Tensor) -> np.ndarray:
        head=self._head_images(np.stack([np.stack(self.image_history[i][-8:]) for i in indices])); agent=torch.from_numpy(np.stack([self.normalizers.agent(np.stack(self.agent_history[i][-8:])) for i in indices])).to(self.device)
        with torch.enable_grad():
            hc=self.model.head_encoder(head); ac=self.model.agent_encoder(agent); latent=ac; tau=torch.full((len(indices),1),.5,device=self.device)
            def velocity(x,t,h,a,s): return self.model.layer1_velocity(x[None],t[None],h[None],a[None],self.model.state_projection(s[None]))[0]
            jac=torch.func.vmap(torch.func.jacrev(velocity,argnums=4))(latent,tau,hc,ac,applied)
        return torch.linalg.matrix_norm(jac,ord="fro").detach().cpu().numpy()

    def _update_s2(self, indices: Sequence[int], boundary: int) -> None:
        indices = [i for i in indices if should_update_s2(boundary, int(self.tasks[i]["d"]))]
        if not indices: return
        goals = np.stack([self.goals[i] for i in indices]); batch = self._l2_batch(indices, goals, boundary)
        with torch.no_grad():
            identity=(self.model.lewm(batch["current_image"])-self.mu)/self.sigma; target=self._oracle(indices,boundary)
            if self.model.config.hierarchical:
                window=(self.model.lewm(batch["observation_window"])-self.mu)/self.sigma; goal=(self.model.lewm(batch["goal_image"])-self.mu)/self.sigma; condition=self.model.layer2.condition(window,batch["action_history"],goal,batch["history_mask"],batch["remaining_horizon"]); raw,path_length=self.model.layer2.integrate(identity,condition)
            else: raw=(self.model.lewm(batch["goal_image"])-self.mu)/self.sigma; path_length=torch.zeros(raw.shape[0],device=raw.device)
            applied = raw
            source = "self_prediction"
            if self.condition == "oracle": applied = self._oracle(indices, boundary); source = "fixed_demonstration_future"
            elif self.condition == "identity": applied = identity; source = "current_state"
            elif self.condition == "goal-swap-recompute":
                swapped = np.stack([self.goals[self.partners[i]] for i in indices]); applied = self.model.predict_s2(self._l2_batch(indices, swapped, boundary), self.mu, self.sigma); source = "cyclic_goal_recompute"
            elif self.condition == "zero": applied = torch.zeros_like(raw); source = "zero_vector"
            elif self.condition == "flat-direct-goal": source = "direct_goal"
        jacobians=self._jacobian_norms(indices,applied); raw_np, applied_np = raw.cpu().numpy(), applied.cpu().numpy(); identity_np=identity.cpu().numpy(); target_np=target.cpu().numpy(); path_np=path_length.cpu().numpy()
        for local, index in enumerate(indices):
            self.raw_s2[index] = raw_np[local]; self.s2[index] = applied_np[local]
            self.slow_events[index].append({"boundary": boundary, "goal_step": int(self.tasks[index]["d"]), "remaining_horizon": min(16,int(self.tasks[index]["d"])-boundary)/16, "source": source, "raw_norm": float(np.linalg.norm(raw_np[local])), "norm": float(np.linalg.norm(applied_np[local])), "start_norm":float(np.linalg.norm(identity_np[local])),"target_norm":float(np.linalg.norm(target_np[local])),"endpoint_mse":float(np.mean((raw_np[local]-target_np[local])**2)),"start_endpoint_distance":float(np.linalg.norm(identity_np[local]-target_np[local])),"path_length":float(path_np[local]),"jacobian_frobenius":float(jacobians[local]),"jacobian_tau":.5,"goal_swap_distance":float(np.linalg.norm(raw_np[local]-applied_np[local])) if self.condition=="goal-swap-recompute" else None,"raw_s2":raw_np[local].tolist(),"applied_s2":applied_np[local].tolist(),"swap_partner": None if self.condition != "goal-swap-recompute" else self.partners[index]})

    def observe(self, images: np.ndarray, agents: np.ndarray, active: np.ndarray) -> None:
        for i in range(len(self.tasks)):
            if active[i]: self.image_history[i].append(images[i].copy()); self.image_history[i] = self.image_history[i][-16:]; self.agent_history[i].append(agents[i].copy()); self.agent_history[i] = self.agent_history[i][-8:]; self.steps[i] += 1

    def plan(self, active: np.ndarray) -> None:
        indices = [i for i in range(len(self.tasks)) if active[i] and not self.queues[i]]
        if not indices: return
        for boundary in sorted(set(int(self.steps[i]) for i in indices if int(self.steps[i]) % 16 == 0)):
            self._update_s2([i for i in indices if int(self.steps[i]) == boundary], boundary)
        head = self._head_images(np.stack([np.stack(self.image_history[i][-8:]) for i in indices])); agent = torch.from_numpy(np.stack([self.normalizers.agent(np.stack(self.agent_history[i][-8:])) for i in indices])).to(self.device)
        state = torch.from_numpy(np.stack([self.s2[i] for i in indices])).to(self.device)
        with torch.no_grad(): actions = self.model.predict_actions(head, agent, state).cpu().numpy()
        if not np.isfinite(actions).all(): raise RuntimeError("h3_evaluation_blocked: non-finite actions")
        for local,index in enumerate(indices): self.queues[index] = list(actions[local])
        for index in indices: self.chunk_events[index].append({"u":int(self.steps[index]),"held_boundary":held_boundary(int(self.steps[index]),int(self.tasks[index]["d"])),"s2_sha256":hashlib.sha256(np.ascontiguousarray(self.s2[index]).view(np.uint8)).hexdigest()})

    def next_actions(self, active: np.ndarray) -> np.ndarray:
        self.plan(active); output = np.zeros((len(self.tasks),2), np.float32)
        for i in range(len(self.tasks)):
            if active[i]: output[i] = self.queues[i].pop(0); self.action_history[i].append(output[i].copy()); self.action_history[i] = self.action_history[i][-16:]
        return output


def render_goal_image(hdf5: Path, task: Mapping[str, Any]) -> np.ndarray:
    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1"); os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from stable_worldmodel.envs.pusht.env import PushT
    table = h3_data.EpisodeTable.from_hdf5(hdf5)
    with h5py.File(hdf5, "r", swmr=True) as handle:
        goal = np.asarray(handle["state"][table.global_row(int(task["episode_idx"]), int(task["goal_step"]))], dtype=np.float64)
    env = PushT(resolution=224, relative=True, render_mode="rgb_array"); env.reset(seed=None)
    env._set_state(goal.copy()); env._set_goal_state(goal.copy())
    return np.asarray(env.render()).transpose(2, 0, 1)


def run_group(*, model, condition, tasks, hdf5, normalizers, checkpoint_sha256, input_sha256, device):
    table = h3_data.EpisodeTable.from_hdf5(hdf5); keys = set()
    for task in tasks:
        for boundary in range(0, int(task["d"]), 16): keys.add((int(task["episode_idx"]), min(int(task["start_step"])+boundary+16, int(task["goal_step"]))))
    oracle = {}
    with h5py.File(hdf5, "r", swmr=True) as handle:
        for key in keys:
            value = np.asarray(handle["pixels"][table.global_row(*key)])
            oracle[key] = value.transpose(2,0,1) if value.shape[-1] == 3 else value
    H3ClockedBatchPolicy.context = {**H3ClockedBatchPolicy.context, "oracle_images": oracle}
    original = reused.ClockedBatchPolicy; reused.ClockedBatchPolicy = H3ClockedBatchPolicy
    try: records, actions = reused.run_group(model=model, condition=condition, tasks=tasks, hdf5=hdf5, normalizers=normalizers, checkpoint_sha256=checkpoint_sha256, input_sha256=input_sha256, device=device)
    finally: reused.ClockedBatchPolicy = original
    policy = H3ClockedBatchPolicy.last_instance
    assert policy is not None
    for index, record in enumerate(records):
        record["record_type"] = "h3_closed_loop_episode"; record["task_index"]=index; record["slow_state_events"] = policy.slow_events[index]; record["layer1_chunk_events"]=policy.chunk_events[index]; record["goal_swap_partner"] = None if condition != "goal-swap-recompute" else policy.partners[index]; record["slow_state_reference"] = policy.slow_events[index][0]["source"]
    return records, actions


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--condition",choices=CONDITIONS,required=True); p.add_argument("--d",type=int,choices=(25,50,75),required=True); p.add_argument("--evaluation-seed",type=int,choices=(42,43,44),required=True); p.add_argument("--hdf5",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device",default="cpu"); p.add_argument("--mac-preflight",action="store_true"); return p.parse_args()


def main() -> int:
    args=parse_args(); model,meta=h3_model.load_checkpoint(args.checkpoint,args.device)
    if meta["update"] != (8 if args.mac_preflight else 30_000): raise ValueError("checkpoint update does not match the evaluation profile")
    tasks=reused.load_tasks(args.data_root/"evaluation_tasks.jsonl",args.d,args.evaluation_seed,count=1 if args.mac_preflight else 50); normalizers=h3_data.Normalizers.load(args.data_root/"normalizers.npz"); H3ClockedBatchPolicy.context={"s2_normalizer":h3_data.S2Normalizer.load(args.data_root/"s2_normalizer.npz")}
    if args.mac_preflight and args.condition=="goal-swap-recompute":
        donor=reused.load_tasks(args.data_root/"evaluation_tasks.jsonl",args.d,args.evaluation_seed,count=2)[1]
        H3ClockedBatchPolicy.context["swap_donor_goal"]=render_goal_image(args.hdf5,donor)
    records,actions=run_group(model=model,condition=args.condition,tasks=tasks,hdf5=args.hdf5,normalizers=normalizers,checkpoint_sha256=reused.sha256_file(args.checkpoint),input_sha256=reused.sha256_file(args.data_root/"evaluation_tasks.jsonl"),device=torch.device(args.device))
    args.output_root.mkdir(parents=True,exist_ok=True); shard=args.output_root/f"raw_rollouts_{args.condition}_d{args.d}_m{model.config.seed}_e{args.evaluation_seed}.jsonl"; shard.write_text("".join(json.dumps(x)+"\n" for x in records),encoding="utf-8"); np.savez_compressed(shard.with_suffix(".npz"),**actions); event_records=[(record,event) for record in records for event in record["slow_state_events"]]; events=[x[1] for x in event_records]; np.savez_compressed(shard.with_name(shard.stem+"_instrumentation.npz"),condition=np.asarray([args.condition]*len(events)),d=np.asarray([args.d]*len(events)),model_seed=np.asarray([model.config.seed]*len(events)),evaluation_seed=np.asarray([args.evaluation_seed]*len(events)),episode_idx=np.asarray([x[0]["episode_idx"] for x in event_records]),start_step=np.asarray([x[0]["start_step"] for x in event_records]),boundary=np.asarray([x["boundary"] for x in events]),raw_s2=np.asarray([x["raw_s2"] for x in events],np.float32),applied_s2=np.asarray([x["applied_s2"] for x in events],np.float32),jacobian_frobenius=np.asarray([x["jacobian_frobenius"] for x in events]),path_length=np.asarray([x["path_length"] for x in events]),goal_swap_distance=np.asarray([np.nan if x["goal_swap_distance"] is None else x["goal_swap_distance"] for x in events])); return 0


if __name__=="__main__": raise SystemExit(main())
