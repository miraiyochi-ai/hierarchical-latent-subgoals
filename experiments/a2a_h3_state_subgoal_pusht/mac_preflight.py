#!/usr/bin/env python3
"""H3 G3 complete-structure Mac preflight orchestrator."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import torch
import h3_model

def structural_checks(device:torch.device)->dict[str,object]:
 class FakeLeWM(torch.nn.Module):
  def __init__(self): super().__init__(); self.register_buffer("anchor",torch.zeros(1))
  def forward(self,x): return x.float().mean(dim=(-1,-2)).mean(dim=-1,keepdim=True).expand(*x.shape[:-3],192)
 fake=FakeLeWM(); model=h3_model.H3Policy(h3_model.H3ModelConfig(),lewm=fake).to(device).eval(); head=torch.zeros(1,8,3,224,224,device=device); agent=torch.zeros(1,8,2,device=device); slow=torch.randn(1,192,device=device); goal_a=torch.zeros(1,3,224,224,device=device); goal_b=torch.ones_like(goal_a)
 with torch.no_grad(): first=model.predict_actions(head,agent,slow); model.lewm(goal_a); model.lewm(goal_b); second=model.predict_actions(head,agent,slow)
 return {"layer1_goal_bit_invariant":torch.equal(first,second),"goal_named_layer1_parameters":[n for n,_ in model.layer1_velocity.named_parameters() if "goal" in n.lower()],"full_image_size":model.config.image_size==224,"s2_projection":list(model.state_projection.weight.shape)==[512,192]}
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--hdf5",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device"); a=p.parse_args(); g2=a.output_root/"g2_small_gate/verdict.json"
 if not g2.is_file() or json.loads(g2.read_text()).get("verdict")!="h3_g2_small_gate_pass": raise RuntimeError("h3_mac_preflight_blocked: G2 has not passed")
 device=a.device or ("mps" if torch.backends.mps.is_available() else "cpu"); checks=structural_checks(torch.device(device)); env={**os.environ,"CUBLAS_WORKSPACE_CONFIG":":4096:8"}; here=Path(__file__).resolve().parent
 for condition in ("hierarchical","flat-direct-goal"):
  subprocess.run([sys.executable,str(here/"h3_train.py"),"--profile","mac_preflight","--condition",condition,"--hdf5",str(a.hdf5),"--data-root",str(a.data_root),"--output-root",str(a.output_root),"--device",device,"--supply","reference"],check=True,env=env)
 for condition in ("predicted","oracle","identity","goal-swap-recompute","zero","flat-direct-goal"):
  checkpoint=a.output_root/f"g3_mac_preflight/checkpoints/{'flat-direct-goal' if condition=='flat-direct-goal' else 'hierarchical'}_ema.pt"
  for d in (25,50,75): subprocess.run([sys.executable,str(here/"h3_evaluate.py"),"--checkpoint",str(checkpoint),"--condition",condition,"--d",str(d),"--evaluation-seed","42","--hdf5",str(a.hdf5),"--data-root",str(a.data_root),"--output-root",str(a.output_root/"g3_mac_preflight"),"--device",device,"--mac-preflight"],check=True,env=env)
 raw=list((a.output_root/"g3_mac_preflight").glob("raw_rollouts_*.jsonl")); checks["rollouts_18"]=len(raw)==18; records=[json.loads(line) for path in raw for line in path.read_text().splitlines() if line]; checks["clock_hold_contract"]=all(all(e["held_boundary"]==min(16*(e["u"]//16),16*((int(r["d"])-1)//16)) for e in r["layer1_chunk_events"]) for r in records); summaries=[json.loads(path.read_text()) for path in (a.output_root/"g3_mac_preflight/checkpoints").glob("*.summary.json")]; checks["all_trainable_gradients"]=len(summaries)==2 and all(x["all_trainable_parameters_received_nonzero_gradient"] and x["frozen_lewm_gradient_count"]==0 for x in summaries); checks["frozen_sha_unchanged"]=len(summaries)==2 and all(x["lewm_sha_before"]==x["lewm_sha_after"] for x in summaries); checks["passed"]=all(v is True or v==[] for k,v in checks.items() if k!="passed"); result={"verdict":"h3_g3_mac_preflight_pass" if checks["passed"] else "h3_mac_preflight_blocked","checks":checks}; (a.output_root/"g3_mac_preflight/verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result)); return 0 if checks["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
