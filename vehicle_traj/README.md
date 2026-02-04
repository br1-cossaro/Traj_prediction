# `vehicle_traj`: Waymo vehicle trajectory prediction

This subproject mirrors the Hydra-style layout from `Urban_traj/`, but targets
vehicle trajectory prediction using Waymo scenes stored in `.npz` files.

## Requirements

- Python 3.10+ with:
  - `torch`
  - `hydra-core`
  - `omegaconf`

## Quickstart (baseline: XY coords, multi-agent)

From the repo root (where `vehicle_traj/`, `baselines/`, `scripts/` live):

```bash
python vehicle_traj/train.py data=waymo_debug train.epochs=1 device=cpu
```

## Stages (feature flags)

Extra features are wired but disabled by default:

- `inputs.use_velocity=true`
- `inputs.use_heading=true`
- `inputs.use_type=true`
- `inputs.use_map=true`
- `inputs.use_traffic_lights=true`

Note: for these flags to work, the `.npz` files must contain matching keys
(for example `velocity`, `heading`, `agent_type`). The debug dataset only
contains `positions/valid_mask/ego_idx`.

To evaluate a checkpoint:

```bash
python vehicle_traj/evaluate.py eval.checkpoint=/path/to/ckpt.pt
```

## Multimodal (comparison with Scene Transformer)

To produce multiple hypotheses (K modes), set `model.num_modes=6` and train with
`train.w_mode_ce` (classification of the best mode by FDE). The `metrics.csv`
will include `val_minade/val_minfde` in addition to `val_ade/val_fde`.

## Logging and run comparison

`train.py` writes a `metrics.csv` inside the Hydra run dir (configurable with
`TP_RUNS_DIR`), with one row per epoch:

- `epoch, seconds, lr, train_loss/train_ade/train_fde, val_loss/val_ade/val_fde`

To list and compare runs:

```bash
python scripts/summarize_vehicle_traj_runs.py --limit 20
```

## Fixed train/val split (recommended)

For fair comparisons, use the same train/val split across pipelines.

Create the split once (on Grid):
```bash
python scripts/make_womd_split.py \
  --root "$HOME/work_clean/data/processed/womd_v1_3_1_npz" \
  --seed 0 --val-ratio 0.2 \
  --out "$HOME/work_clean/data/splits/womd_seed0_val0.2.npz"
```

Use it in `vehicle_traj`:
```bash
python vehicle_traj/train.py \
  data.split_path="$HOME/work_clean/data/splits/womd_seed0_val0.2.npz"
```

## Qualitative plot (GT vs k=6, baseline comparison)

If you have a multimodal run (`model.num_modes=6`), you can visualize a
validation scene and compare against Scene Transformer:
```bash
export WOMD_NPZ_DIR="$HOME/work_clean/data/processed/womd_v1_3_1_npz"
export WOMD_SPLIT_PATH="$HOME/work_clean/data/splits/womd_seed0_val0.2.npz"

CKPT_ST="$HOME/work_clean/runs/scene_transformer_iclr22_womd_agents_only/2026-01-03_17-53-51/checkpoints/epoch_050.pt"
RUN_VT="$HOME/work_clean/runs/vehicle_traj/2026-01-07/10-18-53"
CKPT_VT="$RUN_VT/checkpoints/epoch_050.pt"

python scripts/plot_qualitative_predictions.py \
  --data-root "$WOMD_NPZ_DIR" \
  --split-path "$WOMD_SPLIT_PATH" --subset val --subset-idx 0 \
  --st-ckpt "$CKPT_ST" \
  --vt-run-dir "$RUN_VT" --vt-ckpt "$CKPT_VT" \
  --device cuda \
  --out "$HOME/work_clean/plots/qual_st_vs_vt_val0.png"
```

## Configs

- Main config: `trajectory_prediction/vehicle_traj/config/config.yaml`
- Debug dataset: `trajectory_prediction/vehicle_traj/config/data/waymo_debug.yaml`
- Baseline model: `trajectory_prediction/vehicle_traj/config/model/vehicle_transformer.yaml`
- Prediction target (all vs ego): `trajectory_prediction/vehicle_traj/config/task/base.yaml`
