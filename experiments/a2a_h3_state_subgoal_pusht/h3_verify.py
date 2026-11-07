#!/usr/bin/env python3
"""Independent H3 artifact verifier; intentionally imports no training/evaluation code."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
CONDITIONS=("predicted","oracle","identity","goal-swap-recompute","zero","flat-direct-goal"); HORIZONS=(25,50,75); SEEDS=(42,43,44); COMPARISONS=(("predicted","zero"),("predicted","identity"),("predicted","goal-swap-recompute"),("predicted","flat-direct-goal"),("oracle","identity"),("oracle","predicted"))
def digest(path):
 d=hashlib.sha256();
 with Path(path).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""): d.update(x)
 return d.hexdigest()
def load_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def paired_bootstrap(delta,seed=999):
 value=np.asarray(delta).reshape(3,50); rng=np.random.default_rng(seed); samples=np.empty(10_000)
 for i in range(10_000): samples[i]=np.concatenate([row[rng.integers(0,50,50)] for row in value]).mean()
 ordered=np.sort(samples); return float(np.mean(delta)),[float(ordered[250]),float(ordered[9749])],samples
def materialize_shards(root):
 rollout=root/"raw_rollouts.jsonl"
 if not rollout.exists():
  shards=sorted(x for x in root.glob("raw_rollouts_*.jsonl") if x.name!="raw_rollouts.jsonl"); rollout.write_text("".join(x.read_text() for x in shards),encoding="utf-8")
 action_path=root/"raw_actions.npz"
 if not action_path.exists():
  values={}
  for path in sorted(root.glob("raw_rollouts_*.npz")):
   if "instrumentation" in path.name: continue
   with np.load(path,allow_pickle=False) as f:
    for key in f.files:
     if key in values: raise RuntimeError("duplicate action key")
     values[key]=f[key]
  np.savez_compressed(action_path,**values)
 inst=root/"instrumentation.npz"
 if not inst.exists():
  merged=defaultdict(list)
  for path in sorted(root.glob("*_instrumentation.npz")):
   with np.load(path,allow_pickle=False) as f:
    for key in f.files: merged[key].append(f[key])
  combined={k:np.concatenate(v) for k,v in merged.items()}; np.savez_compressed(inst,**combined); raw=combined.get("raw_s2",np.empty((0,192))); covariance=np.cov(raw,rowvar=False,ddof=0) if len(raw) else np.empty((0,0)); summary={"shards":len(list(root.glob("*_instrumentation.npz"))),"arrays":{k:list(v.shape) for k,v in combined.items()},"raw_s2_mean":raw.mean(0).tolist() if len(raw) else [],"raw_s2_variance":raw.var(0).tolist() if len(raw) else [],"raw_s2_covariance_eigenvalues":np.linalg.eigvalsh(covariance).tolist() if len(raw) else []}; (root/"instrumentation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
def compute_metrics(root,rows):
 rng=np.random.default_rng(999); indices=rng.integers(0,50,size=(10_000,3,50)); by={(int(x["d"]),int(x["model_seed"]),int(x["episode_idx"]),int(x["start_step"]),x["condition"]):int(x["success"]) for x in rows}; metrics={}; boot={"indices":indices}
 for d in HORIZONS:
  task_keys={seed:sorted({(ep,start) for dd,s,ep,start,c in by if dd==d and s==seed}) for seed in SEEDS}
  for a,b in COMPARISONS:
   delta=np.asarray([[by[(d,seed,ep,start,a)]-by[(d,seed,ep,start,b)] for ep,start in task_keys[seed]] for seed in SEEDS],float); samples=np.asarray([np.concatenate([delta[j,indices[i,j]] for j in range(3)]).mean() for i in range(10_000)]); ordered=np.sort(samples); key=f"d{d}:{a}-{b}"; metrics[key]={"point":float(delta.mean()),"ci95":[float(ordered[250]),float(ordered[9749])],"by_seed":{str(seed):float(delta[j].mean()) for j,seed in enumerate(SEEDS)}}; boot[key]=samples
 np.savez_compressed(root/"bootstrap_samples.npz",**boot); (root/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); return metrics
def scientific_verdict(metrics):
 def superiority(a,b):
  for d,margin in ((50,.03),(75,.05)):
   m=metrics[f"d{d}:{a}-{b}"]
   if m["point"]<margin or m["ci95"][0]<=0 or not all(v>0 for v in m["by_seed"].values()): return False
  return True
 if not superiority("predicted","zero") or not superiority("predicted","goal-swap-recompute"): return "subgoal_not_used"
 if not superiority("predicted","identity"): return "future_state_not_better_than_identity"
 if not superiority("oracle","identity"): return "subgoal_generator_gap"
 for d in (50,75):
  m=metrics[f"d{d}:oracle-predicted"]
  if m["point"]>.05 or m["ci95"][1]>.10:return "subgoal_generator_gap"
 if not superiority("predicted","flat-direct-goal"): return "no_improvement_over_flat"
 return "h3_state_subgoal_pusht_gate_pass"
def verify(root:Path)->dict:
 required=["source_manifest.json","protocol.json","data/s2_normalizer.npz","parameter_manifest.json","training_losses.jsonl","raw_rollouts.jsonl","raw_actions.npz","instrumentation.npz","instrumentation.json"]
 missing=[x for x in required if not (root/x).is_file()]; result={"missing":missing,"passed":False}
 if missing:return result
 gate_files=(("g0",root/"data/input_verification.json","h3_g0_input_pass"),("g1",root/"g1_equivalence/comparison.json","h3_g1_equivalence_pass"),("g2",root/"g2_small_gate/verdict.json","h3_g2_small_gate_pass"),("g3",root/"g3_mac_preflight/verification.json","h3_g3_mac_preflight_pass")); result["gates"]={name:(path.is_file() and json.loads(path.read_text()).get("verdict")==expected) for name,path,expected in gate_files}
 checkpoint_ok=True
 for seed in SEEDS:
  for name in ("hierarchical_ema.pt","flat_direct_goal_ema.pt"):
   path=root/"checkpoints"/f"seed_{seed}"/name; summary=path.with_suffix(".summary.json"); checkpoint_ok &= path.is_file() and summary.is_file() and json.loads(summary.read_text()).get("updates")==30_000
 result["checkpoints_complete"]=bool(checkpoint_ok)
 training=load_jsonl(root/"training_losses.jsonl"); training_counts=Counter((x.get("condition"),x.get("model_seed")) for x in training); result["training_counts"]={str(k):v for k,v in training_counts.items()}; result["training_complete"]=len(training_counts)==6 and all(v==30_000 for v in training_counts.values()) and all(np.isfinite([x["gradient_norm"],*x["losses"].values()]).all() for x in training)
 rows=load_jsonl(root/"raw_rollouts.jsonl"); counts=Counter((x["condition"],int(x["d"]),int(x["model_seed"])) for x in rows); result["episode_count"]=len(rows); result["cell_counts_ok"]=len(counts)==54 and all(v==50 for v in counts.values()); keys=[(x["condition"],x["d"],x["model_seed"],x["episode_idx"],x["start_step"],x["goal_step"]) for x in rows]; result["unique_rows"]=len(keys)==len(set(keys)); result["finite"]=all(np.isfinite([x["executed_steps"],x["final_position_error"],x["final_angle_error"]]).all() for x in rows)
 result["success_recomputed"]=all(bool(x["success"])==(x["termination_reason"]=="environment_terminated" and x.get("success_step") is not None) for x in rows)
 result["clock_ok"]=all(all(e["boundary"]%16==0 and e["boundary"]<int(x["d"]) and e["remaining_horizon"]>0 for e in x["slow_state_events"]) and all(e["held_boundary"]==min(16*(e["u"]//16),16*((int(x["d"])-1)//16)) for e in x["layer1_chunk_events"]) for x in rows)
 swaps=[x for x in rows if x["condition"]=="goal-swap-recompute"]; result["swap_self_matches"]=sum(x.get("goal_swap_partner") in (None,x.get("task_index")) for x in swaps)
 by=defaultdict(dict)
 for x in rows: by[(x["d"],x["model_seed"],x["episode_idx"],x["start_step"])][x["condition"]]=int(x["success"])
 complete_pairs=all(set(v)==set(CONDITIONS) for v in by.values()); result["paired_conditions"]=complete_pairs
 result["passed"]=all(result["gates"].values()) and result["checkpoints_complete"] and result["training_complete"] and len(rows)==2700 and result["cell_counts_ok"] and result["unique_rows"] and result["finite"] and result["success_recomputed"] and result["clock_ok"] and result["swap_self_matches"]==0 and complete_pairs
 if result["passed"]:
  metrics=compute_metrics(root,rows); result["scientific_verdict"]=scientific_verdict(metrics); (root/"verdict.md").write_text(f"# H3 verdict\n\n`{result['scientific_verdict']}`\n",encoding="utf-8")
 return result
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); a=p.parse_args(); materialize_shards(a.root); result=verify(a.root); (a.root/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 if result["passed"]:
  artifacts={str(x.relative_to(a.root)):digest(x) for x in a.root.rglob("*") if x.is_file() and x.name!="DONE.json"}; (a.root/"DONE.json").write_text(json.dumps({"verdict":"h3_g5_evaluation_complete","artifacts":artifacts},indent=2),encoding="utf-8")
 print(json.dumps(result)); return 0 if result["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
