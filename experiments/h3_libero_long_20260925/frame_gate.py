"""Check cached/live encoder parity and restored dataset image orientation."""
from pathlib import Path
import json
import os
import sys
import h5py
import numpy as np
import torch

# Experiment root (the --root given to prepare.py); override with H3_LIBERO_ROOT.
ROOT=Path(os.environ.get('H3_LIBERO_ROOT','/workspace/h3_libero_long'))
from common import atomic_json, digest, progress, shared_root
from encoders import visual_model, encode_images
from evaluate import simulator
from training import configure
from replay_demonstrations import relocate_assets, state_render_indices, compare_observation, restored_frame_pass


def main():
    root=shared_root(ROOT); out=root/'frame_gate_v2'; out.mkdir(exist_ok=False)
    configure(42); progress(out,'initialize',0)
    package,Env=simulator(root,out)
    import robosuite
    encoder=visual_model(root/'models')
    manifest=json.loads((root/'source_prep_v1/input_manifest.json').read_text())
    atomic_json(out/'contract.json',dict(source=digest(Path(__file__)),
        replay_helper_sha256=digest(Path(__file__).with_name('replay_demonstrations.py')),
        tasks=10,demo=0,restored_frames_per_task=3,proprio_max=1e-7,rgb_mae_255=2,
        image_unflipped_must_fit_better=True,feature_relative_l2_max=.03,
        feature_note='BF16 batch-size/attention-kernel tolerance, not bit-identical feature claim'))
    results=[]
    for number,item in enumerate(manifest['files']):
        path=root/'data'/item['path']; slug=path.name.removesuffix('_demo.hdf5')
        with h5py.File(path,'r') as h,h5py.File(root/'features'/path.name,'r') as cache:
            demo=h['data/demo_0']
            frames=np.stack([demo['obs/'+key][0] for key in ('agentview_rgb','eye_in_hand_rgb')])
            current=encode_images(encoder,frames).reshape(2048)
            stored=cache['demo_0/visual'][0]
            relative=float(np.linalg.norm(current-stored)/np.linalg.norm(stored))
            if not np.isfinite(relative) or relative>.03: raise ValueError(f'cached/live feature parity failed: {relative}')
            xml,changes=relocate_assets(demo.attrs['model_file'],Path(robosuite.__file__).parent,package/'assets')
            env=Env(bddl_file_name=str(package/'bddl_files/libero_10'/f'{slug}.bddl'),camera_heights=128,camera_widths=128)
            records=[]
            try:
                env.reset_from_xml_string(xml)
                for index in state_render_indices(len(demo['actions'])):
                    obs=env.set_init_state(demo['states'][index+1])
                    metrics=compare_observation(obs,demo,index)
                    passed=restored_frame_pass(metrics)
                    records.append(dict(index=index,metrics=metrics,passed=passed))
                    if not passed:
                        atomic_json(out/'failed_frame.json',dict(task=number,record=records[-1]))
                        raise ValueError('restored dataset frame mismatch')
            finally: env.close()
            results.append(dict(task=number,file=path.name,feature_relative_l2=relative,frames=records))
            atomic_json(out/'partial.json',dict(results=results))
            progress(out,'verified_tasks',len(results))
    atomic_json(out/'result.json',dict(state='frame_gate_passed',results=results))
    progress(out,'complete',10)


if __name__=='__main__': main()
