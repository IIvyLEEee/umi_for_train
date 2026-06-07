# UMI QAT 自定义算子使用说明

## 1. 精度入口与架构配置

FP32 与 QAT 使用独立 policy：

- FP32：`diffusion_policy.policy.diffusion_transformer_timm_policy.DiffusionTransformerTimmPolicy`
- QAT：`diffusion_policy.policy.diffusion_transformer_timm_qat_policy.DiffusionTransformerTimmQATPolicy`

两份架构配置：

- `train_diffusion_transformer_umi_workspace.yaml`：768 维、8 heads、7 layers。
- `train_diffusion_transformer_umi_workspace_our.yaml`：256 维、4 heads、8 layers。
- `train_diffusion_transformer_umi_workspace_qat.yaml`：基于 768 配置，切换到
  QAT policy，并从指定 768 FP32 checkpoint 初始化。

推荐直接使用独立 QAT 配置：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config-name=train_diffusion_transformer_umi_workspace_qat
```

256 FP32 使用 `train_diffusion_transformer_umi_workspace_our`。

## 2. QAT 模型组织

QAT policy 仅把 action transformer 切换到：

```text
diffusion_policy/module/transformer_for_action_diffusion.py
```

观察编码器、scheduler、训练损失与推理噪声生成方式保持当前
`umi_for_qat` policy 的行为，不引入 `umi_for_train` 的固定种子自定义噪声生成器。

QAT action transformer 的调用链：

```text
action
  -> quant_linear_b input embedding
  -> decoder layers
       -> quant_linear_a Q/K/V
       -> custom Softmax
       -> quant_linear_b attention output
       -> nn.LayerNorm, elementwise_affine=False
       -> quant_linear_a FFN input
       -> custom relu
       -> quant_linear_c FFN output
  -> nn.LayerNorm, elementwise_affine=False
  -> quant_linear_b output head
```

该训练数值路径以 `real_demo/umi_for_train` 为基准。Softmax 保留其分段指数
近似并使用 `torch.sum`；action transformer 的 LayerNorm 均使用
`nn.LayerNorm`，不使用 bit-true 查表加法。

## 3. 256 与 768 路径

### 256

- `n_emb=256`，4 heads，8 layers。
- FFN 为 `256 -> 1024 -> 256`，与 `umi_for_train` 的实际结构一致。
- Decoder 和 final norm 使用无 affine 的 `nn.LayerNorm`。

### 768

- `n_emb=768`，8 heads，7 layers。
- FFN 为 `768 -> 3072 -> 768`。
- Decoder 和 final norm 使用无 affine 的 `nn.LayerNorm`。

Decoder 不硬编码 FFN 宽度，统一使用 `4*n_emb`。因此 256 为 1024，768 为
3072；这也与给定 768 FP32 checkpoint 的 FFN shape 一致。

## 4. 设备选择

所有 Python 文件中的 `cuda:N` 已清除。模块中的中间张量跟随输入或模型设备，
物理 GPU 通过运行命令设置：

```bash
CUDA_VISIBLE_DEVICES=2 python ...
```

进程内配置使用逻辑设备 `cuda` 或 `cuda:0`，不再绑定机器的物理 GPU 编号。

## 5. 查找表依赖

运行所需查找表已复制到仓库根目录：

```text
lookup_tables/int8_tensor.ckpt
lookup_tables/toint_tensor.ckpt
lookup_tables/add_result.ckpt
```

代码通过 `module/lookup_table.py` 按仓库位置解析查找表，不依赖运行时工作目录。
这些运行资产已加入 `.gitignore`，不会误提交到 Git。

- `toint_tensor.ckpt`：当前量化主路径使用，供 `Convert_int8` 模拟芯片
  FP16 到 INT8 转换。
- `int8_tensor.ckpt`：与 `toint_tensor.ckpt` 内容不同，当前主路径未引用，
  作为参考版本保留。
- `add_result.ckpt`：作为参考版本保留，训练代码不加载、不引用。

`eval_for_demo_0628` 顶层没有 `add_result.ckpt`。本地副本来自 demo LayerNorm
实际引用的 `/home/cxz23/new_diffusion/test/add_result.ckpt`，其 SHA256 与
`real_demo/umi_for_train/add_result.ckpt` 完全一致。

模型构造时实际加载的唯一 `.ckpt` 是 `toint_tensor.ckpt`。Quant Linear 使用
自定义 `autograd.Function` 手写 backward，因此反传不经过不可导的查表索引。

## 6. 当前 checkpoint 注意事项

给定 FP32 checkpoint：

```text
/data/archive_liyixuan23/umi_for_train/data/outputs/2025.09.18/16.37.43_train_diffusion_transformer_timm_umi/checkpoints/epoch=0130-train_loss=0.012.ckpt
```

它对应 768 维 FP32 架构。`train_diffusion_transformer_umi_workspace_qat.yaml`
通过 `training.init_checkpoint` 从该 checkpoint 初始化 QAT model 和 EMA，
但不恢复 epoch、global step 或 optimizer；这与普通 `resume` 互斥。

FP32 到 QAT 的显式迁移规则：

- 复制视觉编码器、位置编码、input/head、FFN 等同名同 shape 权重。
- FP32 `in_proj_weight` 拆分为独立 Q/K/V，`out_proj.weight` 映射到
  `fc_out.weight`。
- 忽略 QAT 不存在的 FP32 bias 与 LayerNorm affine 参数。
- 量化 scale 状态不从 FP32 checkpoint 读取，由首次 QAT 前向重新计算。
- 256 架构与 768 checkpoint 的 tensor shape 不兼容。

## 7. 当前训练启动

正式训练使用足够大的 `training.num_epochs=1000000`，由人工停止：

```text
GPU1 / tmux umi_qat768_gpu1 / batch 32 / action LR 1e-4 / no warmup
GPU2 / tmux umi_fp32_256_gpu2 / batch 32
```

默认 batch 试跑结论：

- 768 QAT batch 32、action LR `1e-4`、ViT LR `3e-5` 已完成 100-step
  有限值检查。当前 QAT 配置关闭 2000-step warmup；保留 warmup 时第 0 步
  LR 为 0，下一次前向会稳定复现 scaling factor FP16 溢出。
- 查表索引越界不是 OOM 或 batch size 直接导致。训练过程中部分
  `quant_linear_b.fc_out` 的 scaling factor 转为 FP16 时超过 65504，产生
  Inf；`0 * Inf` 随后产生 NaN。NaN 不会被 `clamp(-130, 130)` 修正，最终在
  `Convert_int8` 中映射成超出 `toint_tensor.ckpt` 范围的索引。
- 256 FP32 batch 64 会超过 48GB 显存；batch 32 已完成单步 smoke test并在
  正式训练中持续运行。

## 8. Batch、Scale 与部署语义

量化层在不同模式下行为不同：

- `model.train()`：每次 forward 根据当前整个 batch 重算
  `input_delta/output_delta`。
- `model.eval()`：复用 checkpoint 中保存的固定 delta，不按推理样本动态重算。

因此 batch1 训练可以避免多个样本共享动态 scale，并缓解 pretrained 映射模型
的训练前向溢出；但它不等于最终部署行为，也不能替代固定-scale eval。当前
pretrained 映射模型即使使用 batch1，在固定-scale eval 中仍于第 8 个样本失败。

随机初始化模型的固定-scale eval 已连续通过 300 个样本，最大 raw scale 约
343，平均 Convert 饱和率约 0.005%。这说明 from-scratch 有可能避开当前
pretrained 权重的范围失配，而且为了避免溢出并不天然要求 batch1；是否正确仍
必须以训练后的固定-scale 部署验证为准。

训练超参数必须区分：

- pretrained QAT 微调候选：action LR `1e-4`，当前不使用 warmup。
- from-scratch：action LR `3e-4`，使用 `2000` 个 optimizer-step warmup。

from-scratch 的 batch1+累积实验中，warmup 计数按 optimizer step，而不是按
microbatch；累计 32 个样本后才进行一次更新和一次 scheduler step。

统一短窗口实验结果：

| 初始化与 batch | 训练窗口 | 训练 Softmax min | 固定-scale eval | eval Softmax min |
|---|---:|---:|---:|---:|
| pretrained, batch1+acc32 | 64 samples | -47.47 | 100 samples | -244 |
| scratch, batch1+acc32 | 64 samples | 正值 | 100 samples | 正值 |
| scratch, batch32 | 64 optimizer steps | 正值 | 100 steps | 正值 |

scratch batch32 的训练 loss 从前10步均值约 1.106 降至末10步约 0.690，
固定-scale eval loss 约 0.54，平均 Convert 饱和率约 0.00041%。当前应优先
长步数验证标准 scratch batch32，而不是因为 pretrained 模型溢出就强制使用
batch1。pretrained batch1 虽未在短窗口产生非有限值，但极端负 Softmax 已违反
attention 概率语义，不能视为正确训练。

## 9. 自定义 Softmax 的负值来源

`real_demo/umi_for_train` 的 action 量化主路径明确使用自定义 `Softmax`，当前
仓库与其主计算一致。该实现不是普通 Softmax，而是对
`x - row_max` 的 `exp(x)` 做分段线性近似。

当前实现存在两个负值来源：

1. 线性最小二乘拟合不保证非负。在 `[-4, -8]` 段，拟合直线约从 `-7.24`
   开始变为负数；在 `[-8, -16]` 段，约从 `-13.77` 开始变为负数。
2. 已构造 16 段系数，最后一段 `[-32, -inf)` 的 `k=0,b=0` 本应将尾区置零，
   但 forward 使用 `range(15)`，第 16 段永远不会执行。小于等于 `-32` 的
   shifted logit 会保留为原始负数，再参与归一化。

最小复现：

```text
input [0, -100]
custom Softmax -> [-0.0101, 1.0098]
torch Softmax  -> [1.0, approximately 0.0]
```

因此极端负 attention 权重不是普通浮点误差。临时通过外部 hook 替换为
`torch.softmax` 后，pretrained batch32 早期纯前向和短训练均保持非负且未出现
原先的立即溢出，说明它很可能切断当前失败链。但直接替换会改变芯片算子模型，
不能在未确认芯片真实尾区行为前作为最终实现。`range(15)` 与已构造但未使用的
第 16 段强烈提示软件实现可能有 bug，应由芯片算子负责人确认。
