# Trajectory Prediction Model (vehicle_traj)

Code snapshot for the **vehicle_traj** pipeline.
This repo only contains the model code + dataset loader required to train/evaluate.

## Contents
- `vehicle_traj/` - training, evaluation, configs, model architectures
- `datasets/` - Waymo NPZ dataset loader

## Quick start
```bash
# env used in grid5000
source ~/work_clean/envs/tpred/bin/activate

# data paths
export WOMD_NPZ_DIR=/path/to/womd_npz
export WOMD_SPLIT_PATH=/path/to/womd_split.npz

# train (example)
python vehicle_traj/train.py \
  device=cuda seed=0 \
  data.root="$WOMD_NPZ_DIR" data.split_path="$WOMD_SPLIT_PATH" \
  data.max_agents=128 data.batch_size=4 data.num_workers=2 \
  inputs.use_velocity=true inputs.use_heading=true inputs.use_type=true \
  inputs.use_map=false inputs.use_traffic_lights=false \
  model=vehicle_transformer model.num_modes=6 \
  train.epochs=50 train.lr=1e-4 train.weight_decay=1e-2 train.w_mode_ce=0.1

# eval (meters)
python vehicle_traj/eval_from_ckpt.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --data-root "$WOMD_NPZ_DIR" --split-path "$WOMD_SPLIT_PATH" \
  --device cuda --amp \
  --prediction-target all_agents \
  --batch-size 1 --num-workers 0
```
