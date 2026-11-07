"""Fixed small simulator-connection diagnostic, without predictor/learning.

Stored states[i] precedes action[i], while stored obs[i] follows that action.
State re-render and open-loop action replay are distinct checks. The latter
cannot be expected bit-exact after restoring only qpos/qvel, not controller or
solver history. Stored dataset rewards/dones are synthetic and are NOT success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import h5py
import numpy as np


TASKS = ('open_the_middle_drawer_of_the_cabinet', 'put_the_bowl_on_the_plate', 'turn_on_the_stove')
DEMOS = (0, 45)
COMMIT = '8f1084e3132a39270c3a13ebe37270a43ece2a01'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def verify_checkout(checkout):
    # Network-volume files retain the prior container's UID. Scope this trust
    # to the one known checkout and this read-only invocation, never global/*.
    command = ['git', '-c', f'safe.directory={checkout}', '-C', str(checkout)]
    if subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('LIBERO revision mismatch')
    if subprocess.check_output(command + ['status', '--porcelain'], text=True).strip():
        raise ValueError('official checkout must be clean')


def relocate_assets(xml, robosuite_root, libero_assets):
    """Only change mesh/texture file paths, preserving all physical attributes."""
    root = ET.fromstring(xml)
    changes = []
    asset = root.find('asset')
    if asset is None:
        raise ValueError('XML assets required')
    for node in asset:
        old = node.get('file')
        if old is None:
            continue
        if node.tag not in ('mesh', 'texture'):
            raise ValueError('unexpected external asset type')
        parts = Path(old).parts
        if 'robosuite' in parts:
            pos = max(i for i, part in enumerate(parts) if part == 'robosuite')
            target = robosuite_root.joinpath(*parts[pos + 1:]).resolve()
            boundary = robosuite_root.resolve()
        elif 'assets' in parts and any(part.lower() in ('libero', 'chiliocosm') for part in parts):
            pos = max(i for i, part in enumerate(parts) if part == 'assets')
            target = libero_assets.joinpath(*parts[pos + 1:]).resolve()
            boundary = libero_assets.resolve()
        else:
            raise ValueError(f'unrecognized saved asset namespace: {old}')
        if not target.is_relative_to(boundary) or not target.is_file():
            raise ValueError(f'missing/out-of-bound asset: {target}')
        node.set('file', str(target))
        changes.append(dict(old=old, new=str(target)))
    return ET.tostring(root, encoding='unicode'), changes


def verify_task_metadata(attrs, task, bddl):
    # Released libero-v1 files have a current bddl_file_name, legacy env_args
    # names, and no bddl_file_content. Do not invent the absent text or require
    # the historical display label to match the current dataset label.
    identifier = str(attrs['bddl_file_name'])
    if Path(identifier).parts[-2:] != ('libero_goal', task + '.bddl'):
        raise ValueError('saved task identifier mismatch')
    body = attrs.get('bddl_file_content')
    if body is not None and body.strip() != bddl.read_text().strip():
        raise ValueError('saved and current BDDL differ')
    return dict(saved_identifier=identifier, current_bddl_sha256=digest(bddl),
                saved_body_present=body is not None,
                saved_body_equality_verified=body is not None,
                mapping='released current bddl_file_name, not legacy env_args aliases')


def state_render_indices(length):
    if length < 4:
        raise ValueError('insufficient demonstration states')
    return (0, (length - 2) // 2, length - 2)


def compare_observation(obs, demo, index):
    result = {}
    for name, stored in (('robot0_joint_pos', 'joint_states'), ('robot0_gripper_qpos', 'gripper_states')):
        current, target = np.asarray(obs[name]), np.asarray(demo[f'obs/{stored}'][index])
        if current.shape != target.shape or not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError('invalid proprioceptive observation')
        result[stored + '_mae'] = float(np.abs(current - target).mean())
        result[stored + '_max'] = float(np.abs(current - target).max())
    for name, stored in (('agentview_image', 'agentview_rgb'), ('robot0_eye_in_hand_image', 'eye_in_hand_rgb')):
        current, target = np.asarray(obs[name]), np.asarray(demo[f'obs/{stored}'][index])
        if current.shape != (128, 128, 3) or target.shape != current.shape or current.dtype != np.uint8 or target.dtype != np.uint8:
            raise ValueError('invalid RGB observation')
        current, target = current.astype(np.float64), target.astype(np.float64)
        result[stored + '_mae_255'] = float(np.abs(current - target).mean())
        result[stored + '_flipped_mae_255'] = float(np.abs(current[::-1] - target).mean())
    return result


def restored_frame_pass(metrics):
    # Registration before seeing data: exact proprioception tolerance; small
    # renderer differences tolerated, flipped images should not fit better.
    return (metrics['joint_states_max'] <= 1e-7 and metrics['gripper_states_max'] <= 1e-7
            and all(metrics[view + '_mae_255'] <= 2. and
                    metrics[view + '_mae_255'] < metrics[view + '_flipped_mae_255']
                    for view in ('agentview_rgb', 'eye_in_hand_rgb')))


def summarize(records):
    expected = {(task, demo) for task in TASKS for demo in DEMOS}
    if len(records) != 6 or {(row['task'], row['demo']) for row in records} != expected:
        raise ValueError('missing/duplicate replay records')
    if any(row['actions_executed'] != row['length'] or len(row['state_renders']) != 3 for row in records):
        raise ValueError('incomplete replay')
    return dict(episodes=6, restored_frames=18, environment_steps=sum(row['actions_executed'] for row in records),
                restored_frames_passed=sum(restored_frame_pass(frame['metrics']) for row in records for frame in row['state_renders']),
                action_replays_ever_success=sum(row['ever_success'] for row in records),
                action_replays_final_success=sum(row['final_success'] for row in records),
                warning='Demonstration replay success is not a learned model success rate.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', default='demonstration_replay_v1')
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(args.output).name != args.output or args.output in ('.', '..'):
        raise ValueError('new output identity required')
    if os.environ.get('MUJOCO_GL') != 'osmesa' or os.environ.get('PYOPENGL_PLATFORM') != 'osmesa':
        raise ValueError('explicit CPU OSMesa rendering required for this diagnostic')
    output = root / 'diagnostics' / args.output
    output.mkdir(parents=True, exist_ok=False)
    checkout = root / 'LIBERO_8f1084e'
    verify_checkout(checkout)
    package = checkout / 'libero/libero'
    config = output / 'libero_config'
    config.mkdir()
    save_json(config / 'config.yaml', dict(benchmark_root=str(package), bddl_files=str(package/'bddl_files'),
        init_states=str(package/'init_files'), assets=str(package/'assets'), datasets=str(root/'data')))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    sys.path.insert(0, str(checkout))
    import mujoco, robosuite
    from robosuite import macros
    from libero.libero.envs import OffScreenRenderEnv
    if (mujoco.__version__, robosuite.__version__, macros.IMAGE_CONVENTION) != ('2.3.7', '1.4.0', 'opengl'):
        raise ValueError('simulation version or image convention mismatch')
    manifest_path = root / 'goal_source_manifest.json'
    if digest(manifest_path) != 'ce53027c6e094e142cf2a11958b923c85232124afa659358e00ea8aaa0613642':
        raise ValueError('input manifest changed')
    manifest = json.loads(manifest_path.read_text())
    by_name = {Path(row['path']).name: row for row in manifest['files']}
    for task in TASKS:
        entry = by_name[task + '_demo.hdf5']
        if digest(root / 'data' / entry['path']) != entry['lfs']['oid']:
            raise ValueError('input bytes changed')
        print('verified', task, flush=True)
    contract = dict(tasks=TASKS, demos=DEMOS, libero_commit=COMMIT, mujoco=mujoco.__version__, robosuite=robosuite.__version__,
                    rendering='CPU OSMesa', image_convention='opengl', image_size=128, control_freq=20,
                    controller='OSC_POSE with OffScreenRenderEnv evaluation defaults',
                    initialization='saved XML plus states[0], no extra dummy actions; first five original steps already pruned',
                    state_render_mapping='states[i+1] -> obs[i]', actions='stored raw action[i] unchanged; not normalized or clipped',
                    state_render_thresholds=dict(proprio_max=1e-7, rgb_mae_255=2., same_orientation_better_than_flipped=True),
                    success='env.check_success only; stored rewards/dones not used',
                    inputs={task: by_name[task + '_demo.hdf5']['lfs']['oid'] for task in TASKS},
                    source_sha256=digest(Path(__file__)),
                    official_source_sha256={name: digest(checkout / name) for name in
                        ('scripts/create_dataset.py', 'libero/libero/envs/env_wrapper.py', 'libero/libero/envs/bddl_base_domain.py')},
                    training=False, model_weights_loaded=False,
                    limitations=['six selected demonstrations only', 'CPU rendering not the GPU EGL performance path',
                                 'saved qpos/qvel omit controller and solver history; action replay need not be bit-exact'])
    save_json(output / 'contract.json', contract)
    started = time.monotonic()
    records = []
    for task_number, task in enumerate(TASKS):
        entry = by_name[task + '_demo.hdf5']
        with h5py.File(root / 'data' / entry['path'], 'r') as handle:
            if handle['data'].attrs['macros_image_convention'] != 'opengl':
                raise ValueError('saved image convention mismatch')
            env_args = json.loads(handle['data'].attrs['env_args'])
            bddl = package / 'bddl_files/libero_goal' / (task + '.bddl')
            task_metadata = verify_task_metadata(handle['data'].attrs, task, bddl)
            for demo_id in DEMOS:
                np.random.seed(159151 + task_number * 100 + demo_id)
                demo = handle[f'data/demo_{demo_id}']
                actions, states = np.asarray(demo['actions']), np.asarray(demo['states'])
                if actions.shape != (len(states), 7) or not np.isfinite(actions).all() or not np.isfinite(states).all() or len(actions) > 600:
                    raise ValueError('invalid or over-budget demonstration')
                xml, assets = relocate_assets(demo.attrs['model_file'], Path(robosuite.__file__).parent, package/'assets')
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
                try:
                    env.reset_from_xml_string(xml)
                    render_records = []
                    for index in state_render_indices(len(actions)):
                        obs = env.set_init_state(states[index + 1])
                        metrics = compare_observation(obs, demo, index)
                        render_records.append(dict(index=index, state_index=index + 1, metrics=metrics, passed=restored_frame_pass(metrics)))
                    # Clear the state-render path before the continuous replay.
                    env.reset_from_xml_string(xml)
                    env.sim.reset()
                    env.set_init_state(states[0])
                    controller = env.robots[0].controller
                    controller_settings = {key: np.asarray(getattr(controller, key)).tolist() for key in
                        ('use_delta', 'input_min', 'input_max', 'output_min', 'output_max', 'kp', 'kd', 'control_freq')}
                    frames = []
                    ever_success = bool(env.check_success())
                    for index, action in enumerate(actions):
                        obs, _, _, _ = env.step(action)
                        success = bool(env.check_success())
                        ever_success |= success
                        # State errors at every step; RGB comparisons at fixed
                        # first, middle and penultimate steps only.
                        frame = dict(index=index, success=success,
                            joint_mae=float(np.abs(np.asarray(obs['robot0_joint_pos']) - demo['obs/joint_states'][index]).mean()),
                            joint_max=float(np.abs(np.asarray(obs['robot0_joint_pos']) - demo['obs/joint_states'][index]).max()))
                        if index + 1 < len(states):
                            frame['next_state_l2'] = float(np.linalg.norm(env.get_sim_state() - states[index + 1]))
                        if index in state_render_indices(len(actions)):
                            frame['observation_metrics'] = compare_observation(obs, demo, index)
                        frames.append(frame)
                        if index == 0 or (index + 1) % 25 == 0:
                            progress = dict(phase='action_replay', completed_episodes=len(records), total_episodes=6,
                                task=task, demo=demo_id, step=index + 1, total_steps=len(actions),
                                elapsed_s=time.monotonic()-started, time=time.time())
                            save_json(output/'progress.json', progress)
                            print(json.dumps(progress), flush=True)
                    record = dict(task=task, demo=demo_id, length=len(actions), actions_executed=len(frames),
                                  state_renders=render_records, frames=frames, ever_success=ever_success, final_success=success,
                                  source_env_args=env_args, task_metadata=task_metadata, current_controller=controller_settings,
                                  saved_xml_sha256=hashlib.sha256(demo.attrs['model_file'].encode()).hexdigest(), asset_relocations=assets)
                    records.append(record)
                    with (output/'episodes.jsonl').open('a') as handle_out:
                        handle_out.write(json.dumps(record, allow_nan=False) + '\n')
                finally:
                    env.close()
    summary = summarize(records)
    save_json(output/'result.json', dict(status='demonstration_replay_complete', summary=summary,
        contract_sha256=digest(output/'contract.json'), episodes_sha256=digest(output/'episodes.jsonl'), elapsed_s=time.monotonic()-started))
    save_json(output/'progress.json', dict(phase='complete', completed_episodes=6, total_episodes=6, time=time.time()))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
