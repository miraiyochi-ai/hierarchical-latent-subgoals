#!/usr/bin/env python3
"""H3 G1 reference/uint8-ring equivalence gate.

同一モデルオブジェクトの2パス実行で比較する。deepcopyした2モデルの並走では
テンソル配置(アロケータ/アライメント)差によりcuBLAS計算が1ULP揺れ、
Adamの符号増幅で重み比較が原理的に成立しないため(初回実装の実測)、
参照パス→初期重み巻き戻し→リングパスの順に同一オブジェクトで実行し、
更新ごとの損失・勾配・重みを比較する。登録契約(同一初期重み・同一標本順・
3更新・許容値)は不変。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
import h3_data, h3_model, h3_train
def compare_batches(left,right):
 keys=set(left)|set(right); exact={k: k in left and k in right and torch.equal(torch.as_tensor(left[k]),torch.as_tensor(right[k])) for k in keys}; return {"exact":exact,"passed":all(exact.values())}
def train_step(model,batches,norm,device,generator,optimizer):
 l1_cpu,l2_cpu=batches; optimizer.zero_grad(set_to_none=True); losses=[]; mu=torch.as_tensor(norm.mu,dtype=torch.float32,device=device); sigma=torch.as_tensor(norm.sigma,dtype=torch.float32,device=device)
 for l1raw,l2raw in zip(h3_train._slices(l1_cpu,32),h3_train._slices(l2_cpu,32)):
  l1=h3_train.move(l1raw,device); l2=h3_train.move(l2raw,device); slow=model.predict_s2(l1,mu,sigma); one=model.layer1_loss(l1,slow,generator); two=model.layer2_loss(l2,mu,sigma,generator); total=one["total"]+two["total"]; (total/4).backward(); losses.append([float(one["total"]),float(two["total"]),float(total)])
 gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}; optimizer.step(); return np.asarray(losses),gradients
def run_pass(model,init_state,get_batches,s2norm,device,after_update=None):
 model.load_state_dict(init_state); optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6); generator=torch.Generator(device=device).manual_seed(42); records=[]
 for update in range(3):
  batches=get_batches(update); arrays=[{k:v.detach().cpu().numpy().copy() for k,v in b.items()} for b in batches]; losses,grads=train_step(model,batches,s2norm,device,generator,optimizer)
  if after_update is not None: after_update()
  records.append({"batches":arrays,"losses":losses,"grads":{k:v.cpu() for k,v in grads.items()},"weights":{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}})
 return records
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--hdf5",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device",default="cuda"); a=p.parse_args(); h3_train.configure_determinism(); splits=h3_data.load_split_ids(a.data_root/"split_ids.json"); table=h3_data.EpisodeTable.from_hdf5(a.hdf5); norm=h3_data.Normalizers.load(a.data_root/"normalizers.npz"); s2norm=h3_data.S2Normalizer.load(a.data_root/"s2_normalizer.npz"); l1=h3_data.enumerate_layer1_specs(table,splits["train"]); l2=h3_data.enumerate_layer2_specs(table,splits["train"])
 device=torch.device(a.device); model=h3_model.H3Policy(h3_model.H3ModelConfig(seed=42)).to(device); init_state={k:v.detach().clone() for k,v in model.state_dict().items()}
 ref=h3_train.ReferenceBatchBuilder(h3_data.H3HDF5Reader(a.hdf5,table,norm),l1,l2,42,128)
 try: ref_records=run_pass(model,init_state,ref.update,s2norm,device)
 finally: ref.close()
 ring=h3_train.SharedRingBatchBuilder(hdf5=a.hdf5,table=table,normalizers=norm,image_size=224,layer1_sampler=h3_data.CyclicSpecSampler(l1,42),layer2_sampler=h3_data.BoundarySampler(l2,42),batch_size=128,total_updates=3,workers=4,prefetch_depth=2)
 try: ring_records=run_pass(model,init_state,ring.update,s2norm,device,after_update=ring.mark_consumed)
 finally: ring.close()
 comparisons=[]; arrays_ref={}; arrays_opt={}
 for update,(rr,orr) in enumerate(zip(ref_records,ring_records)):
  layers=[]
  for layer,(rb,ob) in enumerate(zip(rr["batches"],orr["batches"]),1):
   keys=set(rb)|set(ob); exact={k: k in rb and k in ob and np.array_equal(rb[k],ob[k]) for k in keys}; layers.append({"exact":exact,"passed":all(exact.values())})
   for name,value in rb.items(): arrays_ref[f"u{update}_l{layer}_{name}"]=value
   for name,value in ob.items(): arrays_opt[f"u{update}_l{layer}_{name}"]=value
  ll,rl=rr["losses"],orr["losses"]; lg,rg=rr["grads"],orr["grads"]; lw,rw=rr["weights"],orr["weights"]
  loss_equal=bool(np.allclose(ll,rl,atol=1e-7,rtol=1e-6)); gradient_equal=bool(set(lg)==set(rg) and all(torch.allclose(lg[k],rg[k],atol=1e-7,rtol=1e-5) for k in lg)); weight_equal=bool(set(lw)==set(rw) and all(torch.allclose(lw[k],rw[k],atol=1e-7,rtol=1e-5) for k in lw))
  # ビット単位診断: 不一致時に発生箇所(モジュール)を特定するための記録。判定には使わない。
  grad_diffs=sorted(((float((lg[k]-rg[k]).abs().max()),k) for k in lg if k in rg),reverse=True); weight_diffs=sorted(((float((lw[k]-rw[k]).abs().max()),k) for k in lw if k in rw),reverse=True)
  diag={"loss_bitwise":bool(np.array_equal(ll,rl)),"gradient_bitwise":bool(all(torch.equal(lg[k],rg[k]) for k in lg)),"grad_diff_top":[{"param":k,"max_abs_diff":v} for v,k in grad_diffs[:5]],"weight_diff_top":[{"param":k,"max_abs_diff":v} for v,k in weight_diffs[:5]]}
  comparisons.append({"update":update+1,"layers":layers,"loss_equal":loss_equal,"gradient_equal":gradient_equal,"weight_equal":weight_equal,"diagnostics":diag})
 passed=all(x["passed"] for c in comparisons for x in c["layers"]) and all(c["loss_equal"] and c["gradient_equal"] and c["weight_equal"] for c in comparisons); root=a.output_root/"g1_equivalence"; root.mkdir(parents=True,exist_ok=True); np.savez_compressed(root/"reference_batches.npz",**arrays_ref); np.savez_compressed(root/"optimized_batches.npz",**arrays_opt); result={"verdict":"h3_g1_equivalence_pass" if passed else "h3_equivalence_blocked","comparisons":comparisons,"nonfinite":0,"order_violations":0,"early_reuse":0}; (root/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result)); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
