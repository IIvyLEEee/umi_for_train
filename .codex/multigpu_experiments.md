# Multi-GPU Experiment Commands

All commands run from `/data/archive_liyixuan23/umi_for_qat` in the `umi`
environment. `BATCH_SIZE_PER_GPU` is configurable and means the batch size on
each GPU. The effective global batch is:

```text
BATCH_SIZE_PER_GPU * NUM_PROCESSES * training.gradient_accumulate_every
```

Set the shared launch variables first:

```bash
cd /data/archive_liyixuan23/umi_for_qat

ACCELERATE=/home/liyixuan23/miniforge3/envs/umi/bin/accelerate
GPU_IDS=0,1,3
NUM_PROCESSES=3
BATCH_SIZE_PER_GPU=16
```

The QAT commands below train from scratch. This is intentional: the existing
checkpoint configured in `train_diffusion_transformer_umi_workspace_qat.yaml`
was trained on another dataset, and mapping it directly into QAT previously
caused unstable quantized ranges. The QAT 768 commands therefore override
`training.init_checkpoint=null`, restore the from-scratch learning rate, and
restore warmup.

## example_pick_night

### 1. FP32 our config (256)

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29511 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_our \
  task.dataset_path=example_pick_night/mixdataset.zarr.zip \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_night_fp32_our256
```

### 2. QAT our config (256)

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29512 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_qat_our \
  task.dataset_path=example_pick_night/mixdataset.zarr.zip \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_night_qat_our256
```

### 3. QAT 768 config

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29513 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_qat \
  task.dataset_path=example_pick_night/mixdataset.zarr.zip \
  training.init_checkpoint=null optimizer.lr=3.0e-4 training.lr_warmup_steps=2000 \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_night_qat_768_scratch
```

## example_pick_place

### 1. FP32 768 config

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29521 \
  train.py --config-name=train_diffusion_transformer_umi_workspace \
  task.dataset_path=example_pick_place/pick_place_dataset.zarr.zip \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_place_fp32_768
```

### 2. FP32 our config (256)

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29522 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_our \
  task.dataset_path=example_pick_place/pick_place_dataset.zarr.zip \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_place_fp32_our256
```

### 3. QAT 768 config

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29523 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_qat \
  task.dataset_path=example_pick_place/pick_place_dataset.zarr.zip \
  training.init_checkpoint=null optimizer.lr=3.0e-4 training.lr_warmup_steps=2000 \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_place_qat_768_scratch
```

### 4. QAT our config (256)

```bash
${ACCELERATE} launch --multi_gpu \
  --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" --main_process_port 29524 \
  train.py --config-name=train_diffusion_transformer_umi_workspace_qat_our \
  task.dataset_path=example_pick_place/pick_place_dataset.zarr.zip \
  dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  val_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
  exp_name=pick_place_qat_our256
```

## Run the three example_pick_night experiments sequentially

The following command runs the requested three night experiments one after
another on the same GPU set. Change `BATCH_SIZE_PER_GPU` before launching.

```bash
cd /data/archive_liyixuan23/umi_for_qat
GPU_IDS=0,1,3 NUM_PROCESSES=3 BATCH_SIZE_PER_GPU=16 \
  bash scripts/run_pick_night_multigpu.sh
```
