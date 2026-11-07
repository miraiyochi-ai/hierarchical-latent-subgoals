# Predicted Future Latents as Subgoals

Code for the UniReps 2026 workshop extended abstract
**"Predicted Future Latents as Subgoals: Hierarchical Policies Outperform Flat Policies Beyond the Training Goal Distance"**
(Takahiro Morimoto, Miraiyochi, inc.).

A goal-conditioned imitation policy is split into two levels that run inside one network, with no
test-time search. Every 16 environment steps the upper level generates, in the latent space of a
frozen visual encoder, the latent of the observation it expects 16 steps ahead; the lower level takes
that predicted latent as its subgoal and generates the next 8 actions. Only the upper level sees how
far away the final goal is. The repository contains the training and evaluation code for the three
experiments reported in the paper: the PushT comparison against a parameter-matched flat policy
(training goal distances 25/50/75, evaluation up to 200), the PushT range-extension run that adds
100/150 to the training distances, and the LIBERO-Long (`libero_10`) comparison with frozen V-JEPA 2
image features and frozen T5 language goals.

## Repository layout

The directory names are the ones the scripts were written against; several modules locate their
siblings by name (`h3_*.py` imports `h1_*.py` from `../a2a_h1_clockwork_pusht_gate`, and so on), so
keep the layout as is.

```
experiments/
  a2a_h1_clockwork_pusht_gate/   lower level: data pipeline, flow-matching action generator, PushT evaluator
    h1_data.py h1_model.py h1_train.py h1_evaluate.py
  a2a_h3_state_subgoal_pusht/    PushT main experiment (hierarchical vs. flat, 3 training seeds)
    h3_data.py h3_model.py h3_train.py h3_evaluate.py
    stage_g0.py stage_g1.py stage_g2.py mac_preflight.py   pre-training gates (fixed inputs, supply-path equivalence, small gate, CPU/MPS preflight)
    h3_verify.py                                          independent verifier of the produced artifacts
    test_h3_contracts.py
  a2a_h3x_horizon_extension/     evaluation beyond the training range (d = 100, 150, 200) on the same weights
    h3x_make_tasks.py h3x_make_tasks_d200.py h3x_evaluate.py
  h3_pusht_train150_20260926/    range extension: retrain with d in {25,50,75,100,150}, evaluate, paired comparison
    train150.py evaluate150.py compare_results.py test_*.py
  h3_libero_long_20260925/       LIBERO-Long with frozen V-JEPA 2 + T5
    prepare.py preflight.py train.py evaluate.py pipeline.py frame_gate.py replay_demonstrations.py
    common.py data.py encoders.py model.py training.py temporal_contract.py test_*.py
requirements-pusht.txt           package versions used for the PushT runs
requirements-libero.txt          package versions used for the LIBERO-Long runs
```

No `PYTHONPATH` is needed: each script inserts the sibling directories it imports from.

## Installation

### PushT

The PushT runs used Python 3.10, CUDA (H100), and the versions in `requirements-pusht.txt`:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-pusht.txt
```

`stable-worldmodel` (pinned to commit `15a5538d`) provides the PushT environment, state restoration
and success test. `h5py` is pulled in by `stable-worldmodel` and was not pinned separately.

### LIBERO-Long

The LIBERO runs used Python 3.11 with torch 2.9.1 / torchvision 0.24.1 from the base image and the
versions in `requirements-libero.txt`. LIBERO itself is used from a clean checkout of commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`, placed at `<root>/LIBERO_8f1084e` (see below);
`evaluate.py` refuses any other revision. Off-screen rendering uses EGL (`MUJOCO_GL=egl`).

```bash
python3.11 -m venv .venv-libero
.venv-libero/bin/pip install torch==2.9.1 torchvision==0.24.1
.venv-libero/bin/pip install -r requirements-libero.txt
git clone --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git <root>/LIBERO_8f1084e
git -C <root>/LIBERO_8f1084e checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

## External inputs (not redistributed here)

| Input | Source | Used by |
|---|---|---|
| PushT expert demonstrations | Hugging Face dataset `quentinll/lewm-pusht`, file `pusht_expert_train.h5.zst` at revision `655cd446b9929369d7d406001da85c15d1457850`; sha256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318` (13,136,247,974 bytes); decompresses to 46,300,921,856 bytes | all PushT scripts |
| Frozen LeWorldModel-type encoder (ViT-tiny, 192-D) | the Hi-LeWM artifact on Zenodo, DOI `10.5281/zenodo.21353240` (`package.zip`): code under `package/code` (module `h_le_wm`; it in turn expects the upstream LeWorldModel source in `third_party/lewm`, see its `THIRD_PARTY_LEWM.md`) and checkpoint `package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`, sha256 `b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008` (122,424,517 bytes) | `h3_model.py` (upper level) |
| LIBERO-Long demonstrations | Hugging Face dataset `yifengzhu-hf/LIBERO-datasets`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, the 10 `libero_10/*.hdf5` files (sizes and sha256 in `experiments/h3_libero_long_20260925/input_manifest.json`) | `prepare.py` downloads them |
| V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256`, revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | `encoders.py` |
| T5-base | `google-t5/t5-base`, revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` | `encoders.py` |

By default `h3_model.py` looks for the Hi-LeWM artifact under `third_party/hi-lewm/{code,checkpoints}`
in this repository; set `H3_LEWM_CODE` and `H3_LEWM_CHECKPOINT` to point elsewhere.

## PushT

### Data preparation

Decompress the dataset (`zstd -d pusht_expert_train.h5.zst`) and keep the `.zst` file: `stage_g0.py`
checks its sha256. The PushT scripts also take a `--data-root` directory holding four small fixed
inputs: `split_ids.json` (train/validation/test episode split), `evaluation_tasks.jsonl` (the
in-range start/goal pairs, 50 per distance and seed), `normalizers.npz` (image and action
statistics) and `s2_normalizer.npz` (per-component statistics of the frozen encoder's latents over
the training frames, written by `stage_g0.py`). These files are on the Hugging Face Hub under
`pusht_data/` (see "Pretrained weights" below).

### Gates before training

```bash
cd experiments/a2a_h3_state_subgoal_pusht
python stage_g0.py --compressed-hdf5 .../pusht_expert_train.h5.zst --hdf5 .../pusht_expert_train.h5 --data-root DATA --output-root OUT --device cuda   # fixed inputs, replay audit, s2 normalizer
python stage_g1.py --hdf5 ... --data-root DATA --output-root OUT --device cuda                                                           # reference vs. optimized supply path equivalence
python stage_g2.py --hdf5 ... --data-root DATA --output-root OUT --device cuda                                                           # small gate
python mac_preflight.py --hdf5 ... --data-root DATA --output-root OUT                                                                    # 8-update end-to-end preflight on CPU/MPS
```

### Training

One run per (condition, seed); 30,000 updates, batch 128, final EMA weights, no checkpoint selection.

```bash
python h3_train.py --profile gpu_full --condition hierarchical     --seed 42 --hdf5 ... --data-root DATA --output-root OUT --device cuda
python h3_train.py --profile gpu_full --condition flat-direct-goal --seed 42 --hdf5 ... --data-root DATA --output-root OUT --device cuda
```

Repeat for seeds 43 and 44. The flat policy is the lower network with its width adjusted so that the
trainable parameter count matches the hierarchical policy within 0.1% (`h3_model.build_capacity_matched_pair`),
receiving the final-goal latent in the slot where the predicted future latent enters.
Checkpoints are written to `OUT/checkpoints/seed_<seed>/{hierarchical,flat_direct_goal}_ema.pt`.

### Evaluation within the training range (d = 25, 50, 75)

```bash
python h3_evaluate.py --checkpoint OUT/checkpoints/seed_42/hierarchical_ema.pt     --condition predicted        --d 50 --evaluation-seed 42 --hdf5 ... --data-root DATA --output-root OUT/evaluation_shards --device cuda
python h3_evaluate.py --checkpoint OUT/checkpoints/seed_42/flat_direct_goal_ema.pt --condition flat-direct-goal --d 50 --evaluation-seed 42 --hdf5 ... --data-root DATA --output-root OUT/evaluation_shards --device cuda
```

Conditions: `predicted` (the hierarchical policy), `flat-direct-goal`, and the diagnostics `identity`
(upper output replaced by the current latent), `oracle` (replaced by the ground-truth future latent),
`zero`, `goal-swap-recompute`. Each cell is 50 trials; the paper reports 3 seeds x 50 per distance.
`h3_verify.py --root OUT` re-checks the produced artifacts independently.

### Evaluation beyond the training range (d = 100, 150, 200)

```bash
cd experiments/a2a_h3x_horizon_extension
export H3X_H5=.../pusht_expert_train.h5 H3X_H3_RESULTS=OUT H3X_OUT=OUT_H3X
python h3x_make_tasks.py          # 150 pairs per distance from the 439 held-out demonstrations (d = 100, 150)
python h3x_make_tasks_d200.py     # same for d = 200
python h3x_evaluate.py --condition predicted        --d 150 --seed 42 --device cuda
python h3x_evaluate.py --condition flat-direct-goal --d 150 --seed 42 --device cuda
```

The same weights are used without retraining; the action budget is 2d.

### Range extension (training on d in {25, 50, 75, 100, 150})

```bash
cd experiments/h3_pusht_train150_20260926
python train150.py --condition hierarchical     --hdf5 ... --data-root DATA --output-root OUT150/hierarchical --device cuda
python train150.py --condition flat-direct-goal --hdf5 ... --data-root DATA --output-root OUT150/flat --device cuda
python evaluate150.py --help      # one evaluation cell (condition, d) with progress reporting; wraps h3_evaluate / h3x_evaluate unchanged
python compare_results.py --old-hierarchical OUT --old-flat OUT --new-hierarchical OUT150/hierarchical --new-flat OUT150/flat
```

`train150.py` reuses the H3 trainer and model unchanged and only widens the training horizon mixture.
It verifies the sha256 of the source files it reuses; two of those files (`h3_model.py`,
`h3x_evaluate.py`) differ in this public release from the run-time originals only in their default
input locations, and the original hashes are kept as comments next to the new ones.

## LIBERO-Long

All large inputs and outputs live under one experiment root (`--root`), which must be inside the shared
directory `/workspace` (a mounted volume in the original runs) or inside `H3_SHARED_ROOT` if that
variable is set.

```bash
cd experiments/h3_libero_long_20260925
python prepare.py  --root ROOT --quota-gb 150 --image-batch 32    # download the 10 libero_10 files, audit post-action timing, cache V-JEPA 2 / T5 features
python preflight.py --root ROOT                                   # parameter-matched configs, gradient/resume gate (writes ROOT/model_configs.json)
python frame_gate.py                                              # H3_LIBERO_ROOT=ROOT; cached-vs-live encoder parity and restored-frame check
python train.py    --root ROOT --architecture hierarchical --name hier_30k --until 30000
python train.py    --root ROOT --architecture flat         --name flat_30k --until 30000
python evaluate.py --root ROOT --checkpoint ROOT/runs/hier_30k/ema_030000.pt --output hier_eval --states 50 --task-indices 0 1 2 3 4 5 6 7 8 9
python pipeline.py --root ROOT --name full_v1                      # the fixed 5k/15k/30k train-then-evaluate sequence used for the paper
```

Per task, demonstrations 0-44 train and 45-49 are held out for diagnostics; evaluation is 10 tasks x
50 initial states with a budget of 520 actions. `pipeline.py` expects a passed `preflight_v1` gate and
the two one-episode smoke evaluations (`evaluate.py --smoke --states 1`) before it starts.

## Pretrained weights

Weights and the small fixed inputs are published at
<https://huggingface.co/Miraiyochi/hierarchical-latent-subgoals>:

| Path on the Hub | Contents |
|---|---|
| `pusht_main_reproduction/seed_42/{hierarchical,flat_direct_goal}_ema.pt` | PushT main setting (training d in {25, 50, 75}), 30k updates, final EMA; seed-42 rerun, see the note below |
| `pusht_train150/seed_42/{hierarchical,flat_direct_goal}_ema.pt` | PushT range extension (training d in {25, 50, 75, 100, 150}), seed 42; the weights reported in the paper |
| `pusht_data/{split_ids.json, evaluation_tasks.jsonl, normalizers.npz, s2_normalizer.npz, evaluation_tasks_extended.jsonl, evaluation_tasks_d200.jsonl}` | fixed inputs for `--data-root` (first four) and the beyond-range tasks (`H3X_OUT/data/`) |
| `libero_long/{hierarchical,flat}/ema_030000.pt` | LIBERO-Long, seed 42, 30k updates; the weights reported in the paper |

Each folder has a `SHA256SUMS` file.

**Note on the PushT main-setting weights.** The paper's main PushT result averages three training
seeds (42, 43, 44). Those six checkpoints were stored on machines without persistent storage and
were not kept. `pusht_main_reproduction` contains a later rerun of seed 42 with the same code,
inputs and settings. Its hierarchical-minus-flat gaps beyond the training range point the same way
as the paper's (+24 points at d = 100 and +26 at d = 150, against +14 and +14 for the original
seed 42), but the numbers are not identical. To reproduce the three-seed table, train seeds 42-44
with the commands above.

## Paper

```bibtex
@inproceedings{morimoto2026futurelatents,
  title     = {Predicted Future Latents as Subgoals: Hierarchical Policies Outperform Flat Policies Beyond the Training Goal Distance},
  author    = {Morimoto, Takahiro},
  booktitle = {UniReps: Unifying Representations in Neural Models (Workshop), Extended Abstract Track},
  year      = {2026},
  note      = {arXiv:TBD}
}
```

## License

MIT. See `LICENSE`. The external inputs listed above keep their own licenses.
