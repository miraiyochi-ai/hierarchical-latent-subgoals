#!/usr/bin/env python3
"""Mechanical tests for every H3 prohibition that can be checked without full data."""
from __future__ import annotations
import inspect
from pathlib import Path
import h5py, numpy as np, pytest, torch
import h3_data, h3_evaluate, h3_model, h3_train, h3_verify, stage_g1, stage_g2

def table(length=101): return h3_data.EpisodeTable(np.asarray([length]),np.asarray([0]))

def test_layer2_teacher_never_crosses_goal():
 pools=h3_data.enumerate_layer2_specs(table(),[0]); specs=[x for rows in pools.values() for x in rows]
 assert specs and all(x.boundary<x.d and x.teacher_step<=x.goal_step for x in specs)
 assert {x.teacher_step-x.task_start_step for x in specs if x.d==25 and x.boundary==16}=={25}

def test_layer1_has_boundary_prediction_identifier_and_two_chunk_hold():
 specs=h3_data.enumerate_layer1_specs(table(),[0])[75]
 task=[x for x in specs if x.task_start_step==0]
 assert [(x.u,x.boundary) for x in task[:4]]==[(0,0),(8,0),(16,16),(24,16)]
 assert task[0].s2_id==task[1].s2_id and task[2].s2_id==task[3].s2_id

def test_layer1_teacher_actions_fit_before_goal():
 for values in h3_data.enumerate_layer1_specs(table(),[0]).values(): assert all(x.u+8<=x.d for x in values)

def test_t0_padding_and_history_mask(tmp_path):
 path=tmp_path/"tiny.h5"
 with h5py.File(path,"w") as f:
  f["ep_len"]=np.asarray([101]); f["ep_offset"]=np.asarray([0]); f["pixels"]=np.arange(101*4*4*3,dtype=np.uint8).reshape(101,4,4,3); f["action"]=np.arange(202,dtype=np.float32).reshape(101,2); f["proprio"]=np.zeros((101,2),np.float32)
 norm=h3_data.Normalizers(np.zeros(2,np.float32),np.ones(2,np.float32)); reader=h3_data.H3HDF5Reader(path,table(),norm)
 zero=reader.layer2(h3_data.Layer2Spec(0,10,25,0)); later=reader.layer2(h3_data.Layer2Spec(0,10,25,16)); reader.close()
 assert torch.equal(zero["observation_window"],zero["current_image"].expand_as(zero["observation_window"])); assert torch.count_nonzero(zero["action_history"])==0; assert zero["history_mask"].item()==0
 assert later["history_mask"].item()==1 and torch.count_nonzero(later["action_history"])>0

def test_no_boundary_or_future_leak():
 l1=[x for rows in h3_data.enumerate_layer1_specs(table(),[0]).values() for x in rows]; l2=[x for rows in h3_data.enumerate_layer2_specs(table(),[0]).values() for x in rows]
 assert h3_data.validate_no_leak(l1,l2)=={"goal_overrun":0,"boundary_overrun":0,"wrong_s2_id":0,"episode_crossing":0}
 for spec in l2:
  refs=h3_data.layer2_reference_steps(spec); assert max(refs["observations"])<=spec.task_start_step+spec.boundary; assert not refs["actions"] or max(refs["actions"])<spec.task_start_step+spec.boundary

def test_horizon_batch_rotation():
 assert h3_data.balanced_horizon_counts(0)=={25:43,50:43,75:42}; assert h3_data.balanced_horizon_counts(1)=={25:42,50:43,75:43}; assert sum(h3_data.balanced_horizon_counts(2).values())==128

def test_s2_normalization_is_identical_for_numpy_and_torch():
 norm=h3_data.S2Normalizer(np.arange(192,dtype=np.float64),np.arange(192,dtype=np.float64)+1,10); x=np.arange(192,dtype=np.float32)*2
 assert np.array_equal(norm.apply(x),norm.apply(torch.from_numpy(x)).numpy()); assert np.allclose(norm.invert(norm.apply(x)),x)

def test_zero_is_exact_standardized_mean():
    norm=h3_data.S2Normalizer(np.random.default_rng(1).normal(size=192),np.ones(192),5); assert np.array_equal(norm.apply(norm.mu),np.zeros(192,np.float32))

def test_layer2_action_normalization_is_fixed_and_shared():
 low=h3_data.ACTION_MINIMUM; high=h3_data.ACTION_MAXIMUM; assert np.array_equal(h3_data.normalize_action(low),low); assert np.array_equal(h3_data.normalize_action(high),high)

def test_official_preprocess_contract():
 x=torch.zeros(2,3,10,12,dtype=torch.uint8); y=h3_model.official_preprocess(x); assert y.shape==(2,3,224,224); expected=torch.tensor([-0.485/0.229,-0.456/0.224,-0.406/0.225]); assert torch.allclose(y[:,:,0,0],expected.expand(2,3))

def test_layer2_endpoint_and_flow_loss_and_six_steps():
 layer=h3_model.StateSubgoalGenerator(); batch=3; x0=torch.randn(batch,192); x1=torch.randn(batch,192); condition=layer.condition(torch.randn(batch,16,192),torch.randn(batch,16,2),torch.randn(batch,192),torch.ones(batch,1),torch.ones(batch,1)); losses=layer.losses(x0,x1,condition,torch.full((batch,1),.5)); assert torch.equal(losses["total"],losses["flow"]+losses["endpoint"]); assert losses["prediction"].shape==(batch,192); assert layer.integrate(x0,condition)[0].shape==(batch,192)

def test_layer1_has_no_goal_argument_or_parameter():
 parameters=inspect.signature(h3_model.H3Layer1VelocityField.forward).parameters; assert "goal" not in parameters; model=h3_model.H3Layer1VelocityField(16); assert not [n for n,_ in model.named_parameters() if "goal" in n.lower()]; assert model.condition_dim==1536

def test_joint_training_does_not_detach_predicted_s2():
 source=inspect.getsource(h3_train.train_one); line=next(x for x in source.splitlines() if "slow = model.predict_s2" in x); assert "detach" not in line; assert "first[\"total\"] + LAYER2_WEIGHT * second[\"total\"]" in source

def test_goal_time_and_after_holds_last_boundary():
 assert h3_evaluate.held_boundary(0,25)==0; assert h3_evaluate.held_boundary(24,25)==16; assert h3_evaluate.held_boundary(25,25)==16; assert h3_evaluate.held_boundary(49,25)==16; assert not h3_evaluate.should_update_s2(32,25)

def test_goal_swap_fixed_cycle_has_no_self_match():
 tasks=[{"task_sha256":str(i),"episode_idx":i,"start_step":0} for i in range(50)]; partners=h3_evaluate.fixed_cyclic_partners(tasks); assert len(set(partners))==50; assert all(i!=p for i,p in enumerate(partners))

def test_reference_optimized_batch_comparison_is_exact():
 batch={"pixels":torch.arange(8,dtype=torch.uint8),"action":torch.ones(2)}; result=stage_g1.compare_batches(batch,{k:v.clone() for k,v in batch.items()}); assert result["passed"]

def test_g2_noise_is_sample_stable_and_bootstrap_indices():
 ids=["a","b","c"]; assert np.array_equal(stage_g2.fixed_noise(ids),stage_g2.fixed_noise(ids)); raw,ci=stage_g2.bootstrap(np.arange(20,dtype=float)); assert len(raw)==10_000 and ci[0]<=ci[1]

def test_independent_verifier_refuses_incomplete_run(tmp_path):
 result=h3_verify.verify(tmp_path); assert not result["passed"] and "raw_rollouts.jsonl" in result["missing"]

def test_required_evaluation_conditions_are_exact():
 assert h3_evaluate.CONDITIONS==("predicted","oracle","identity","goal-swap-recompute","zero","flat-direct-goal")

def test_checkpoint_and_constants_are_registered():
 assert h3_model.S2_DIM==192 and h3_model.EULER_STEPS==6 and h3_data.L2_CLOCK==16 and h3_train.UPDATES==30_000 and h3_train.MICROBATCH_SIZE==32
