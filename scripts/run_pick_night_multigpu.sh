#!/usr/bin/env bash
set -euo pipefail

ACCELERATE="${ACCELERATE:-/home/liyixuan23/miniforge3/envs/umi/bin/accelerate}"
GPU_IDS="${GPU_IDS:-0,1,3}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"

common_args=(
  task.dataset_path=example_pick_night/mixdataset.zarr.zip
  "dataloader.batch_size=${BATCH_SIZE_PER_GPU}"
  "val_dataloader.batch_size=${BATCH_SIZE_PER_GPU}"
)

"${ACCELERATE}" launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29511 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_our \
  "${common_args[@]}" exp_name=pick_night_fp32_our256

"${ACCELERATE}" launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29512 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_qat_our \
  "${common_args[@]}" exp_name=pick_night_qat_our256

"${ACCELERATE}" launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29513 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_qat \
  "${common_args[@]}" \
  training.init_checkpoint=null optimizer.lr=3.0e-4 training.lr_warmup_steps=2000 \
  exp_name=pick_night_qat_768_scratch
