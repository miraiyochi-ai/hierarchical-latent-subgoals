#!/usr/bin/env python3
"""H3 G2 paired bootstrap verdict utilities and registered runner boundary."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np, torch
import h3_data, h3_model, h3_train
CONDITIONS=("state_flow","noise_flow","state_direct","noise_direct")
def fixed_noise(sample_ids,seed=42):
 out=[]
 for value in sample_ids:
  import hashlib
  s=int(hashlib.sha256(f"h3-g2:{seed}:{value}".encode()).hexdigest()[:16],16); out.append(np.random.default_rng(s).standard_normal(192).astype(np.float32))
 return np.stack(out)
def bootstrap(values,seed=999,repeats=10_000):
 rng=np.random.default_rng(seed); n=len(values); raw=np.asarray([np.mean(values[rng.integers(0,n,n)]) for _ in range(repeats)]); ordered=np.sort(raw); return raw,(float(ordered[250]),float(ordered[9749]))
def bootstrap_ratio(numerator,denominator,seed=999,repeats=10_000):
 rng=np.random.default_rng(seed); n=len(numerator); raw=np.empty(repeats)
 for i in range(repeats):
  index=rng.integers(0,n,n); raw[i]=np.mean(numerator[index])/np.mean(denominator[index])
 ordered=np.sort(raw); return raw,(float(ordered[250]),float(ordered[9749]))
def verdict(arrays):
 sf, nf, sd=np.asarray(arrays["state_flow_mse"]),np.asarray(arrays["noise_flow_mse"]),np.asarray(arrays["state_direct_mse"]); _,ci_direct=bootstrap_ratio(sf,sd); _,ci_noise=bootstrap(sf-nf); _,ci_distance=bootstrap(np.asarray(arrays["state_start_distance"])-np.asarray(arrays["noise_start_distance"])); finite=all(np.isfinite(v).all() for v in arrays.values()); passed=finite and sf.mean()/sd.mean()<=1.05 and ci_direct[1]<=1.10 and sf.mean()/nf.mean()<=.95 and ci_noise[1]<0 and np.mean(arrays["state_start_distance"])<np.mean(arrays["noise_start_distance"]) and ci_distance[1]<0
 return {"verdict":"h3_g2_small_gate_pass" if passed else "h3_small_gate_blocked","finite":finite,"state_flow_over_direct":float(sf.mean()/sd.mean()),"state_flow_over_noise":float(sf.mean()/nf.mean()),"ratio_direct_ci":ci_direct,"flow_difference_ci":ci_noise,"start_distance_difference_ci":ci_distance}
def encode_batch(raw,encoder,norm,device):
 mu=torch.as_tensor(norm.mu,dtype=torch.float32,device=device); sigma=torch.as_tensor(norm.sigma,dtype=torch.float32,device=device); to=lambda name:(encoder(raw[name].to(device))-mu)/sigma
 return {"window":to("observation_window"),"x0":to("current_image"),"x1":to("teacher_image"),"goal":to("goal_image"),"actions":raw["action_history"].to(device),"mask":raw["history_mask"].to(device),"remaining":raw["remaining_horizon"].to(device),"ids":[f"{int(e)}:{int(s)}:{int(d)}:{int(b)}" for e,s,d,b in zip(raw["episode_idx"],raw["task_start_step"],raw["d"],raw["boundary"])]}
def train_registered(hdf5,data_root,output_root,device):
 h3_train.configure_determinism(); splits=h3_data.load_split_ids(data_root/"split_ids.json"); table=h3_data.EpisodeTable.from_hdf5(hdf5); reader=h3_data.H3HDF5Reader(hdf5,table,h3_data.Normalizers.load(data_root/"normalizers.npz")); pools=h3_data.enumerate_layer2_specs(table,splits["train"]); sampler=h3_data.BoundarySampler(pools,42); norm=h3_data.S2Normalizer.load(data_root/"s2_normalizer.npz"); encoder=h3_model.FrozenLeWMEncoder().to(device).eval(); models={name:(h3_model.StateSubgoalGenerator() if name.endswith("flow") else h3_model.DirectSubgoalGenerator()).to(device) for name in CONDITIONS}
 # Common condition encoders start identically across all four conditions.
 base=models["state_flow"]; common={k:v for k,v in base.state_dict().items() if k.startswith(("window_encoder.","action_encoder."))}
 for model in models.values(): model.load_state_dict(common,strict=False)
 optimizers={k:torch.optim.AdamW(v.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6) for k,v in models.items()}; generator=torch.Generator(device=device).manual_seed(42)
 for update in range(2000):
  specs=sampler.take_update(update,128); raw=h3_data.collate([reader.layer2(x) for x in specs]); encoded=encode_batch(raw,encoder,norm,device); tau=torch.rand((128,1),device=device,generator=generator)
  for name,model in models.items():
   x0=encoded["x0"] if name.startswith("state") else torch.from_numpy(fixed_noise(encoded["ids"])).to(device); condition=model.condition(encoded["window"],encoded["actions"],encoded["goal"],encoded["mask"],encoded["remaining"]); opt=optimizers[name]; opt.zero_grad(set_to_none=True)
   if name.endswith("flow"): loss=model.losses(x0,encoded["x1"],condition,tau)["total"]
   else: loss=torch.nn.functional.mse_loss(model(x0,condition),encoded["x1"])
   if not torch.isfinite(loss): raise RuntimeError("h3_small_gate_blocked: non-finite loss")
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); factor=(update+1)/500 if update<500 else .5*(1+np.cos(np.pi*(update-499)/1500)); [g.update(lr=1e-4*factor) for g in opt.param_groups]; opt.step()
 root=output_root/"g2_small_gate"; (root/"checkpoints").mkdir(parents=True,exist_ok=True)
 for name,model in models.items(): torch.save({"condition":name,"updates":2000,"state_dict":model.state_dict()},root/"checkpoints"/f"{name}.pt")
 validation_pools=h3_data.enumerate_layer2_specs(table,splits["validation"]); ordered={key:h3_data.fixed_hash_order(value,f"h3-g2-validation-20260810:{key}") for key,value in validation_pools.items()}; categories=sorted(ordered); positions={key:0 for key in categories}; selected=[]
 while len(selected)<4096:
  key=categories[len(selected)%len(categories)]; selected.append(ordered[key][positions[key]]); positions[key]+=1
 arrays={k:[] for k in ("state_flow_mse","noise_flow_mse","state_direct_mse","noise_direct_mse","state_start_distance","noise_start_distance")};
 for n in (1,2,4,6): arrays[f"flow_mse_steps_{n}"]=[]; arrays[f"path_length_steps_{n}"]=[]
 for start in range(0,len(selected),128):
  raw=h3_data.collate([reader.layer2(x) for x in selected[start:start+128]]); e=encode_batch(raw,encoder,norm,device); noise=torch.from_numpy(fixed_noise(e["ids"])).to(device); arrays["state_start_distance"].extend((e["x0"]-e["x1"]).norm(dim=-1).cpu().numpy()); arrays["noise_start_distance"].extend((noise-e["x1"]).norm(dim=-1).cpu().numpy())
  with torch.no_grad():
   for name,model in models.items():
    x=e["x0"] if name.startswith("state") else noise; c=model.condition(e["window"],e["actions"],e["goal"],e["mask"],e["remaining"]); pred=model.integrate(x,c,6)[0] if name.endswith("flow") else model(x,c); arrays[f"{name}_mse"].extend((pred-e["x1"]).square().mean(-1).cpu().numpy())
   flow=models["state_flow"]; c=flow.condition(e["window"],e["actions"],e["goal"],e["mask"],e["remaining"])
   for n in (1,2,4,6): pred,path=flow.integrate(e["x0"],c,n); arrays[f"flow_mse_steps_{n}"].extend((pred-e["x1"]).square().mean(-1).cpu().numpy()); arrays[f"path_length_steps_{n}"].extend(path.cpu().numpy())
 reader.close(); arrays={k:np.asarray(v) for k,v in arrays.items()}; root.mkdir(parents=True,exist_ok=True); np.savez_compressed(root/"validation_predictions.npz",**arrays); return arrays
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--validation-predictions",type=Path); p.add_argument("--hdf5",type=Path); p.add_argument("--data-root",type=Path); p.add_argument("--device",default="cpu"); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args();
 if a.validation_predictions:
  with np.load(a.validation_predictions,allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
 else:
  if a.hdf5 is None or a.data_root is None: p.error("training requires --hdf5 and --data-root")
  arrays=train_registered(a.hdf5,a.data_root,a.output_root,torch.device(a.device))
 required={"state_flow_mse","noise_flow_mse","state_direct_mse","noise_direct_mse","state_start_distance","noise_start_distance","flow_mse_steps_1","flow_mse_steps_2","flow_mse_steps_4","flow_mse_steps_6","path_length_steps_1","path_length_steps_2","path_length_steps_4","path_length_steps_6"}; missing=required-set(arrays)
 result={"verdict":"h3_small_gate_blocked","missing":sorted(missing)} if missing else verdict(arrays); root=a.output_root/"g2_small_gate"; root.mkdir(parents=True,exist_ok=True); (root/"verdict.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result)); return 0 if result["verdict"]=="h3_g2_small_gate_pass" else 2
if __name__=="__main__": raise SystemExit(main())
