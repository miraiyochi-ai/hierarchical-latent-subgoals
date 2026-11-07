#!/usr/bin/env python3
"""H3 G0 fixed-input, leakage, replay, and s2-normalizer gate."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
import h5py, numpy as np, torch
import h3_data, h3_model

EXPECTED={"split_ids.json":"9fb8235254654cb8e8ebf481809fe0084bcb31afaacdae009b2a79627e3afc35","evaluation_tasks.jsonl":"6b347a6416000e8b35352ef72594b71d6e78d9a833edb7c91e393afcdfe7b318","normalizers.npz":"105c53f553f87a06f538bc259ca0f9fc9a79875d051c0bb704f5e7cfc70ccf25"}
# SHA照合済み登録HDF5の実構造(results/.../replay_gate_correction_basis.md 測定3)
EXPECTED_DATASETS={"action":((2_336_736,2),"float32"),"ep_len":((18_685,),"int32"),"ep_offset":((18_685,),"int64"),"episode_idx":((2_336_736,),"int64"),"pixels":((2_336_736,224,224,3),"uint8"),"proprio":((2_336_736,4),"float32"),"state":((2_336_736,7),"float32"),"step_idx":((2_336_736,),"int64")}
# 再生許容値の根拠は replay_gate_correction_basis.md 測定2・4。高速接触課題の再生誤差は
# 二峰性(稀に負対照帯へ届く)のため課題単位の全数一致では分離できず、集約判定とする:
# 個別基準(状態L∞≤4.0+画素ペア判別+画素≤3.0)を44/48以上が満たし、状態誤差中央値≤1.0。
REPLAY_STATE_TOLERANCE=4.0
REPLAY_PIXEL_MEAN_TOLERANCE=3.0
REPLAY_MIN_PASS=44
REPLAY_MEDIAN_STATE_TOLERANCE=1.0
def sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""): d.update(x)
 return d.hexdigest()
def create_normalizer(hdf5:Path, ids:np.ndarray, encoder:h3_model.FrozenLeWMEncoder, output:Path, device:torch.device, batch_size:int=128)->dict:
 table=h3_data.EpisodeTable.from_hdf5(hdf5); count=0; mean=np.zeros(192); m2=np.zeros(192); encoder.to(device).eval()
 with h5py.File(hdf5,"r",swmr=True) as f:
  for ep in map(int,ids):
   rows=np.arange(table.offsets[ep],table.offsets[ep]+table.lengths[ep]);
   for start in range(0,len(rows),batch_size):
    value=np.asarray(f["pixels"][rows[start:start+batch_size]]); value=value.transpose(0,3,1,2) if value.shape[-1]==3 else value
    z=encoder(torch.from_numpy(value).to(device)).cpu().numpy().astype(np.float64)
    for row in z:
     count+=1; delta=row-mean; mean+=delta/count; m2+=delta*(row-mean)
 sigma=np.sqrt(m2/count)
 if count==0 or not np.isfinite(mean).all() or not np.isfinite(sigma).all() or np.any(sigma<=0): raise RuntimeError("h3_input_blocked: invalid latent statistics")
 output.parent.mkdir(parents=True,exist_ok=True); np.savez(output,mu=mean,sigma=sigma,count=np.int64(count),split_sha256=EXPECTED["split_ids.json"],encoder_sha256=h3_model.module_state_sha256(encoder),preprocess=np.asarray("uint8->float/255->ImageNetNormalize->Resize224->encoder(interpolate_pos_encoding=True)->CLS->projector"))
 return {"count":count,"finite":True,"positive_sigma":True}
def replay_48(hdf5:Path,tasks_path:Path)->dict:
 os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1"); os.environ.setdefault("SDL_VIDEODRIVER","dummy")
 from stable_worldmodel.envs.pusht.env import PushT
 tasks=h3_data.load_tasks(tasks_path); selected=[]
 for d in (25,50,75): selected.extend(sorted((x for x in tasks if int(x["d"])==d),key=lambda x:x["task_sha256"])[:16])
 mismatches=[]; records=[]
 with h5py.File(hdf5,"r",swmr=True) as f:
  offsets=np.asarray(f["ep_offset"])
  for task in selected:
   row=int(offsets[int(task["episode_idx"])]+int(task["start_step"])); start=np.asarray(f["state"][row],np.float64); goal=np.asarray(f["state"][row+int(task["d"])],np.float64); env=PushT(resolution=224,relative=True,render_mode="rgb_array"); env.reset(seed=None); env._set_state(start.copy()); env._set_goal_state(goal.copy())
   for action in np.asarray(f["action"][row:row+16],np.float32): env.step(action)
   observed_state=np.asarray(env._get_obs(),np.float64); observed_pixels=np.asarray(env.render()); env.close(); expected_state=np.asarray(f["state"][row+16],np.float64)
   def _hwc(value): return value.transpose(1,2,0) if value.shape[0]==3 else value
   expected_pixels=_hwc(np.asarray(f["pixels"][row+16])); start_pixels=_hwc(np.asarray(f["pixels"][row]))
   # 修正基準(replay_gate_correction_basis.md): 厳密一致はfloat32保存丸め・_set_stateの
   # 余分な物理ステップ・別レンダリングスタックで原理不可能。状態L∞許容+画素ペア判別へ訂正。
   state_error=float(np.max(np.abs(observed_state-expected_state)))
   mean_to_target=float(np.mean(np.abs(observed_pixels.astype(np.int16)-expected_pixels.astype(np.int16)))); mean_to_start=float(np.mean(np.abs(observed_pixels.astype(np.int16)-start_pixels.astype(np.int16))))
   ok=state_error<=REPLAY_STATE_TOLERANCE and mean_to_target<=REPLAY_PIXEL_MEAN_TOLERANCE and mean_to_target<mean_to_start
   if not ok: mismatches.append(task["task_sha256"])
   records.append({"task_sha256":task["task_sha256"],"state_error":state_error,"pixel_mean_to_target":mean_to_target,"pixel_mean_to_start":mean_to_start,"passed":ok})
 median_state_error=float(np.median([x["state_error"] for x in records]))
 return {"count":len(selected),"mismatch_count":len(mismatches),"mismatches":mismatches,"state_tolerance":REPLAY_STATE_TOLERANCE,"pixel_mean_tolerance":REPLAY_PIXEL_MEAN_TOLERANCE,"min_pass":REPLAY_MIN_PASS,"median_state_tolerance":REPLAY_MEDIAN_STATE_TOLERANCE,"median_state_error":median_state_error,"records":records}
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--compressed-hdf5",type=Path,required=True); p.add_argument("--hdf5",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device",default="cpu"); a=p.parse_args(); gates={}
 gates["fixed_sha"] = sha(a.compressed_hdf5)=="7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318" and all(sha(a.data_root/name)==value for name,value in EXPECTED.items()) and sha(h3_model.LEWM_CHECKPOINT)==h3_model.LEWM_SHA256 and h3_model.LEWM_CHECKPOINT.stat().st_size==122424517
 splits=h3_data.load_split_ids(a.data_root/"split_ids.json"); table=h3_data.EpisodeTable.from_hdf5(a.hdf5); l1=h3_data.enumerate_layer1_specs(table,splits["train"]); l2=h3_data.enumerate_layer2_specs(table,splits["train"]); l1_all=[x for v in l1.values() for x in v]; l2_all=[x for v in l2.values() for x in v]; violations=h3_data.validate_no_leak(l1_all,l2_all); gates["no_leak"]=not any(violations.values())
 encoder=h3_model.FrozenLeWMEncoder(); before=h3_model.module_state_sha256(encoder); sample=torch.zeros(1,3,224,224,dtype=torch.uint8); first=encoder(sample); second=encoder(sample); config=encoder.encoder.config; gates["lewm_config"]=int(config.image_size)==224 and int(config.patch_size)==14 and int(config.hidden_size)==192 and encoder.output_dim()==192; gates["frozen_encoder"]=all(not p.requires_grad for p in encoder.parameters()) and not encoder.training and torch.equal(first,second)
 normalizer=create_normalizer(a.hdf5,splits["train"],encoder,a.output_root/"data/s2_normalizer.npz",torch.device(a.device)); gates["normalizer"]=normalizer["finite"] and before==h3_model.module_state_sha256(encoder)
 (a.output_root/"data").mkdir(parents=True,exist_ok=True)
 for name in ("split_ids.json","evaluation_tasks.jsonl"): shutil.copyfile(a.data_root/name,a.output_root/"data"/name)
 np.savez_compressed(a.output_root/"data/boundary_specs.npz",l1_episode=np.asarray([x.episode_idx for x in l1_all],np.int32),l1_task_start=np.asarray([x.task_start_step for x in l1_all],np.int32),l1_d=np.asarray([x.d for x in l1_all],np.int16),l1_u=np.asarray([x.u for x in l1_all],np.int16),l1_boundary=np.asarray([x.boundary for x in l1_all],np.int16),l2_episode=np.asarray([x.episode_idx for x in l2_all],np.int32),l2_task_start=np.asarray([x.task_start_step for x in l2_all],np.int32),l2_d=np.asarray([x.d for x in l2_all],np.int16),l2_boundary=np.asarray([x.boundary for x in l2_all],np.int16),l2_teacher=np.asarray([x.teacher_step for x in l2_all],np.int32))
 replay=replay_48(a.hdf5,a.data_root/"evaluation_tasks.jsonl"); gates["environment_replay_48"]=replay["count"]==48 and (replay["count"]-replay["mismatch_count"])>=REPLAY_MIN_PASS and replay["median_state_error"]<=REPLAY_MEDIAN_STATE_TOLERANCE
 with h5py.File(a.hdf5,"r",swmr=True) as f: shapes={name:list(f[name].shape) for name in f}; gates["hdf5_shape_type_count"]=set(f.keys())==set(EXPECTED_DATASETS) and all(tuple(f[name].shape)==shape and str(f[name].dtype)==dtype for name,(shape,dtype) in EXPECTED_DATASETS.items())
 result={"verdict":"h3_g0_input_pass" if all(gates.values()) else "h3_input_blocked","gates":gates,"leak_violations":violations,"normalizer":normalizer,"replay":replay,"hdf5_shapes":shapes,"encoder_sha256":before}; path=a.output_root/"data/input_verification.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result)); return 0 if all(gates.values()) else 2
if __name__=="__main__": raise SystemExit(main())
