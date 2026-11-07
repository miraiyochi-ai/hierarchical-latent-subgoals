"""Language-only LIBERO-Long closed-loop evaluation with H3 16/8 clocks."""
from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from common import atomic_json, digest, progress, shared_root, source_hashes
from data import Store, limits, inverse_actions
from encoders import visual_model, encode_images
from model import Config, Policy
from training import configure


def proprio(obs):
    result = np.concatenate([obs['robot0_joint_pos'], obs['robot0_gripper_qpos']]).astype('float32')
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError('invalid simulator proprioception')
    return result


def images(obs):
    result = [np.asarray(obs[k]).copy() for k in ('agentview_image','robot0_eye_in_hand_image')]
    if any(x.shape != (128,128,3) or x.dtype != np.uint8 for x in result):
        raise ValueError('invalid simulator images')
    return result


def simulator(root, output):
    checkout = root/'LIBERO_8f1084e'
    commit = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if commit != '8f1084e3132a39270c3a13ebe37270a43ece2a01' or subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True).strip():
        raise ValueError('LIBERO source must match the fixed clean revision')
    cfg = output/'libero_config'; cfg.mkdir()
    package = checkout/'libero/libero'
    atomic_json(cfg/'config.yaml',dict(benchmark_root=str(package), bddl_files=str(package/'bddl_files'),
                init_states=str(package/'init_files'),assets=str(package/'assets'),datasets=str(root/'data')))
    os.environ['LIBERO_CONFIG_PATH'] = str(cfg)
    os.environ['MUJOCO_GL'] = 'egl'; os.environ['PYOPENGL_PLATFORM'] = 'egl'
    sys.path.insert(0,str(checkout))
    import mujoco, robosuite
    from robosuite import macros
    from libero.libero.envs import OffScreenRenderEnv
    if mujoco.__version__!='2.3.7' or robosuite.__version__!='1.4.0' or macros.IMAGE_CONVENTION!='opengl':
        raise ValueError('simulator or image convention mismatch')
    return package, OffScreenRenderEnv


def batch_from_history(z, states, actions, text, norm):
    z = np.asarray(z); states=np.asarray(states)
    visual = ((z-np.asarray(norm['visual_mean']))/np.asarray(norm['visual_std'])).astype('float32')
    state = limits(states,norm['state_min'],norm['state_max']).astype('float32')
    if len(actions)==0:
        action=np.zeros((16,7),'float32'); mask=np.zeros(1,'float32')
    elif len(actions)==16:
        action=limits(np.asarray(actions),norm['action_min'],norm['action_max']).astype('float32'); mask=np.ones(1,'float32')
    else:
        # The upper is only recomputed at 0,16,...; a partial action history is not valid there.
        action=np.zeros((16,7),'float32'); mask=np.zeros(1,'float32')
    values=dict(lower_visual=visual[-8:],lower_states=state[-8:],upper_visual=visual,
                upper_actions=action,upper_mask=mask,remaining=np.ones(1,'float32'),language=text)
    return {k:torch.from_numpy(v).unsqueeze(0).cuda() for k,v in values.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--states',type=int,default=5)
    p.add_argument('--max-steps',type=int,default=520)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--task-indices',type=int,nargs='+')
    a=p.parse_args(); root=shared_root(a.root)
    if Path(a.output).name!=a.output or a.output in ('.','..') or not 1<=a.states<=50 or a.max_steps<1:
        raise ValueError('invalid evaluation identity/budget')
    out=root/'evaluations'/a.output; out.mkdir(parents=True,exist_ok=False)
    configure(42)
    progress(out,'load',0)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if a.smoke:
        if saved['checkpoint_kind']!='full_training_state':
            raise ValueError('smoke expects the explicit preflight snapshot')
        norm=Store(root).normalization; state=saved['ema']
    else:
        if saved['checkpoint_kind']!='evaluation_only' or saved.get('weights_kind')!='ema':
            raise ValueError('evaluation requires an immutable EMA snapshot')
        sidecar=json.loads(a.checkpoint.with_suffix('.json').read_text())
        if sidecar['sha256']!=digest(a.checkpoint):
            raise ValueError('checkpoint integrity mismatch')
        norm=saved['contract']['normalization']; state=saved['model']
    model=Policy(Config(**saved['config'])).cuda().eval(); model.load_state_dict(state)
    language_path=root/'language.json'; language=json.loads(language_path.read_text())
    prepared=json.loads((root/'prepared.json').read_text())
    if digest(language_path)!=prepared['language_sha256']:
        raise ValueError('language integrity mismatch')
    encoder=visual_model(root/'models')
    package, Env=simulator(root,out)
    selected=[0] if a.smoke else (a.task_indices if a.task_indices is not None else list(range(10)))
    if len(set(selected))!=len(selected) or any(t not in range(10) for t in selected):
        raise ValueError('invalid task subset')
    tasks=[(t,language['tasks'][t]) for t in selected]
    max_steps=min(a.max_steps,16) if a.smoke else a.max_steps
    contract=dict(checkpoint_sha256=digest(a.checkpoint),step=saved['step'],seed=42,
        architecture='hierarchical' if model.layer2 is not None else 'flat',tasks=[x['file'] for _,x in tasks],task_indices=selected,
        states=a.states,max_steps=max_steps,smoke=a.smoke,success='env.check_success',
        startup_steps=10,goal='official language only; no goal image or remaining ground-truth duration',
        image_convention='opengl',device=torch.cuda.get_device_name(0),source_sha256=source_hashes(Path(__file__).parent),
        time_includes='V-JEPA observation encoding plus policy; environment time separately',clock=[16,8])
    atomic_json(out/'contract.json',contract)
    results=[]
    with (out/'episodes.jsonl').open('x') as records:
        for task_id,task in tasks:
            slug=task['file'].removesuffix('_demo.hdf5')
            init_path=package/'init_files/libero_10'/f'{slug}.pruned_init'
            initial=torch.load(init_path,map_location='cpu',weights_only=False)
            if len(initial)<a.states:
                raise ValueError('insufficient fixed initial states')
            text=np.asarray(task['embedding'],dtype='float32')
            for state_id in range(a.states):
                env=Env(bddl_file_name=str(package/'bddl_files/libero_10'/f'{slug}.bddl'),camera_heights=128,camera_widths=128)
                try:
                    seed=42+task_id*1000+state_id
                    env.seed(seed); env.reset(); obs=env.set_init_state(np.asarray(initial[state_id]))
                    for _ in range(10): obs,_,_,_=env.step(np.zeros(7,dtype='float32'))
                    torch.cuda.synchronize(); tick=time.monotonic()
                    first=encode_images(encoder,np.stack(images(obs))).reshape(2048)
                    torch.cuda.synchronize(); encoding_time=time.monotonic()-tick
                    z=deque([first.copy() for _ in range(16)],maxlen=16)
                    s=deque([proprio(obs).copy() for _ in range(16)],maxlen=16)
                    acts=deque(maxlen=16)
                    steps,calls,upper_calls,policy_time,environment_time=0,0,0,0.,0.
                    success=bool(env.check_success()); condition=None
                    while steps<max_steps and not success:
                        batch=batch_from_history(z,s,acts,text,norm)
                        torch.cuda.synchronize(); tick=time.monotonic()
                        with torch.inference_mode():
                            if model.layer2 is None: condition=batch['language']
                            elif steps%16==0:
                                condition=model.predict_subgoal(batch); upper_calls+=1
                            predicted=model.actions(batch,condition).cpu().numpy()[0]
                        torch.cuda.synchronize(); policy_time+=time.monotonic()-tick; calls+=1
                        if not np.isfinite(predicted).all(): raise FloatingPointError('invalid policy actions')
                        action_chunk=inverse_actions(predicted,norm)
                        new_images,new_states=[],[]
                        for action in action_chunk:
                            tick=time.monotonic(); obs,_,_,_=env.step(action); environment_time+=time.monotonic()-tick
                            acts.append(action.copy()); steps+=1
                            new_images.extend(images(obs)); new_states.append(proprio(obs))
                            success=bool(env.check_success())
                            if success or steps>=max_steps: break
                        if not success and steps<max_steps:
                            torch.cuda.synchronize(); tick=time.monotonic()
                            encoded=encode_images(encoder,np.stack(new_images)).reshape(len(new_states),2048)
                            torch.cuda.synchronize(); encoding_time+=time.monotonic()-tick
                            z.extend(encoded); s.extend(new_states)
                        progress(out,'rollout',len(results)*1000+steps,completed=len(results),task=task_id,state=state_id,steps=steps)
                    row=dict(task=task_id,file=task['file'],initial_state=state_id,success=success,steps=steps,
                             policy_calls=calls,upper_calls=upper_calls,encoding_seconds=encoding_time,
                             policy_seconds=policy_time,environment_seconds=environment_time,seed=seed)
                    records.write(json.dumps(row)+'\n'); records.flush(); results.append(row)
                finally:
                    env.close()
    reread=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
    expected={(t,i) for t in selected for i in range(a.states)}
    if len(reread)!=len(expected) or {(r['task'],r['initial_state']) for r in reread}!=expected:
        raise ValueError('evaluation coverage failure')
    atomic_json(out/'result.json',dict(state='smoke_complete' if a.smoke else 'evaluation_complete',
        count=len(reread),successes=sum(r['success'] for r in reread),success_rate=np.mean([r['success'] for r in reread]),
        records_sha256=digest(out/'episodes.jsonl'),checkpoint_sha256=contract['checkpoint_sha256']))
    progress(out,'complete',len(results))


if __name__=='__main__':
    main()
