# Progress Log

## 2026-06-06 11:38:16 +0800

- 收到任务：阅读芯片算子模型，说明模型组织形式和 QAT 使用方式。
- 确认用户所指 `module/` 实际路径为 `diffusion_policy/module/`。
- 检查 `diffusion_policy/module/` 的模型入口、基础算子、量化 Linear、转换查找表和运行依赖。
- 检查当前 FP32 policy、训练 workspace、Hydra 配置和 checkpoint 保存/恢复逻辑。
- 确认给定 FP32 checkpoint 存在，大小约 2.8 GB，配置使用 `vit_large_patch14_clip_224.openai` 和 7 层 action diffusion transformer。
- 确认当前训练 policy 仍使用 `diffusion_policy/model/diffusion/transformer_for_action_diffusion.py`，尚未接入 `diffusion_policy/module/`。
- 确认 `module/transformer_for_action_diffusion.py` 当前使用未量化的自定义 `Linear`；真正量化的 `quant_linear_a/b/c.py` 尚未接入完整主干。
- 确认量化算子固定使用 `cuda:2`，并依赖 `/home/cxz23/new_diffusion/eval_for_demo_0628/toint_tensor.ckpt`；当前机器和文件满足该约束。
- 梳理 FP32 MHA 到芯片显式 Q/K/V attention 的权重映射，以及 bias、GELU、LayerNorm affine 等结构差异。
- 创建 `.codex/qat_module_guide.md`，记录模型组织、现状、checkpoint 迁移、推荐 QAT 流程、验收项和风险。
- 未修改 `diffusion_policy/module/` 中任何文件。
- 当前进度：说明文档已完成，等待后续 `rule.md` 维护任务或 QAT 接入实现任务。

## 2026-06-06 11:40:30 +0800

- 完成文档验证：`git diff --check` 无格式错误。
- 确认 checkpoint 路径、Q/K/V 权重映射、`cuda:2` 约束、EMA 初始化建议和量化算子未接入现状均已写入说明文档。
- 确认本次变更仅新增 `.codex/qat_module_guide.md` 和 `.codex/prog.md`，`diffusion_policy/module/` 保持不变。
- 当前进度：本次说明文档任务完成。

## 2026-06-06 12:00:15 +0800

- 搜索 `/home/cxz23/real_demo` 下的全部 `module/` 实现，发现三个版本：`universal_manipulation_interface`、`umi_for_train` 和 `umi_for_demo`。
- 确认 `umi_for_train` 的实际 policy 导入量化 `module/transformer_for_action_diffusion.py`；其 input/head、attention Q/K/V/output projection 和 FFN 均实际调用 `quant_linear_a/b/c`，适合作为 QAT 接入参考。
- 确认 `umi_for_train` 的 attention 使用自定义 `Softmax`，但 LayerNorm 仍使用无 affine 的 `nn.LayerNorm`。
- 确认 `umi_for_demo` 使用量化 self-attention、量化 cross-attention、自定义 Softmax 和自定义 LayerNorm，芯片算子覆盖更完整。
- 发现 `umi_for_demo/module/decoder.py` 在 decoder 第三层执行 `exit()`，属于逐层芯片对齐/调试版本，当前不能作为完整推理主路径直接使用。
- 发现 `umi_for_train` 和 `umi_for_demo` 的 FFN 隐藏维度分别硬编码为 1024 和 768，均不同于原始 FP32 模型的 `4 * n_emb = 3072`，权重迁移时必须处理结构不兼容。

## 2026-06-06 12:16:58 +0800

- 对比 `/home/cxz23/real_demo` 下除 `umi_backup` 外三个项目的标准 UMI、`_our` 和 bimanual Transformer 配置。
- 确认标准单臂配置均声明 `n_emb=768`、`n_head=8`、`n_layer=7`、action horizon 16、action dim 10、执行 action steps 8；每个 attention head 维度为 96。
- 确认 `_our` 配置在三个项目中完全相同，声明 `n_emb=256`、`n_head=4`、`n_layer=8`，每个 attention head 维度为 64。
- 确认 bimanual 配置在三个项目中完全相同，声明 `n_emb=768`、`n_head=8`、`n_layer=7`、action horizon 16、action dim 20。
- 确认标准配置的视觉骨干不同：`universal_manipulation_interface` 使用 ViT-L/14，`umi_for_train` 和 `umi_for_demo` 使用 ViT-B/16。
- 确认配置传入的 FFN 宽度为 `4*n_emb`，但实际 module 实现不同：universal 使用传入值，train 硬编码 1024，demo 硬编码 768。

## 2026-06-06 12:20:49 +0800

- 对比当前 `umi_for_qat` 与 `/home/cxz23/real_demo/umi_for_train` 的标准 Transformer YAML、task 配置、policy 入口和 module 实际结构。
- YAML 有效参数差异：视觉骨干为 ViT-L/14 对 ViT-B/16，训练/验证 batch size 为 32 对 64，当前项目新增 `load_optimizer_on_resume=False`，数据集路径不同。
- 核心声明维度一致：`n_emb=768`、`n_head=8`、`n_layer=7`、action dim 10、action horizon 16、inference steps 16。
- 关键实际结构差异：当前 policy 使用 FP32 model，当前 module 的 FFN 为 3072；`umi_for_train` policy 使用量化 module，FFN 硬编码为 1024。

## 2026-06-06 12:23:00 +0800

- 更正上一轮比较对象：按用户要求，将当前 `umi_for_qat` 标准配置与 `real_demo/umi_for_train` 的 `train_diffusion_transformer_umi_workspace_our.yaml` 对比。
- 核心模型差异：当前为 `n_emb=768`、8 heads、7 layers、head dim 96、FFN 3072；`our` 为 `n_emb=256`、4 heads、8 layers、head dim 64、量化 module FFN 1024。
- 两者均使用 ViT-L/14、action dim 10、action horizon 16、inference steps 16，但 encoder 冻结、batch size、训练设备和 resume 行为不同。
- 确认 `umi_for_train` 的量化 module 硬编码 FFN 1024 正好匹配 `our` 配置的 `4*256`，说明该量化主路径是按 256 维 `our` 模型组织的，不能直接承接当前 768 维 FP32 checkpoint。

## 2026-06-06 16:26:45 +0800

- 按确认方案开始整合 `real_demo/umi_for_train` 的量化 action transformer 调用链。
- 同步 quant Linear A/B/C、multi-head attention、自定义 Softmax、转换器、decoder 和 action transformer。
- 新增独立 QAT policy；现有 FP32 policy 保持不变。
- 引入独立 256 `our` 配置，设置 `training.resume=False` 和 `logging.resume=False`。
- 将 `umi_for_demo` 的 16x256 自定义 LayerNorm 融入 QAT 路径，仅做设备适配和形状校验。
- 新增独立 16x768 LayerNorm 机械扩展版本：24 个 32 元素块，均值系数 `1/768`。
- Decoder FFN 改为使用 `dim_feedforward=4*n_emb`；256 路径保持 1024，768 路径使用 3072。
- 清除仓库所有 Python 文件中的 `cuda:N`，设备由输入/model 和 `CUDA_VISIBLE_DEVICES` 决定。
- 静态语法检查通过；256/768 QAT 结构构造检查通过；两个 LayerNorm 的替代查表形状前向检查通过。
- 确认固定依赖 `/datadisk/umi_for_train/add_result.ckpt` 和 `/datadisk/umi_for_train/toint_tensor.ckpt` 当前不存在；按约束保留原路径，真实 QAT 运行前需提供。
- 更新 `.codex/qat_module_guide.md`，新增 `.codex/integration_change_log.md` 逐文件记录全部改动。
- 当前进度：实现完成，正在执行最终静态检查与验证。

## 2026-06-06 16:30:45 +0800

- 使用仓库推荐的 `/home/liyixuan23/miniforge3/envs/umi/bin/python` 验证 Hydra 可解析现有 FP32 policy 和新增 QAT policy。
- 完成全部改动文件的 `py_compile`/`compileall` 检查。
- 完成 256/768 QAT Transformer 构造检查，确认量化 Linear、Softmax、LayerNorm 类型和 FFN shape 正确。
- 完成 256/768 LayerNorm 的替代查表前向形状检查。
- 完成 `git diff --check`，无格式错误。
- 完成全仓库 Python `cuda:N` 扫描，无残留设备编号硬编码。
- 未执行真实芯片查表前向或完整训练，因为固定依赖 `/datadisk/umi_for_train/add_result.ckpt` 和 `/datadisk/umi_for_train/toint_tensor.ckpt` 当前不存在。
- 当前进度：QAT 入口、算子链、双架构 LayerNorm、配置、设备适配和文档整合全部完成。

## 2026-06-06 16:38:00 +0800

- 核对 `/home/cxz23/new_diffusion/eval_for_demo_0628/`：存在
  `int8_tensor.ckpt` 和 `toint_tensor.ckpt`，项目内部不存在 `add_result.ckpt`。
- 确认 demo LayerNorm 实际从 `/home/cxz23/new_diffusion/test/add_result.ckpt`
  加载加法查找表；该文件与 `real_demo/umi_for_train/add_result.ckpt` SHA256 一致。
- 将三份查找表复制到仓库 `lookup_tables/`，复制后 SHA256 与源文件一致。
- 新增 `module/lookup_table.py`，相关量化模块统一从仓库本地路径加载查找表。
- `toint_tensor.ckpt` 用于当前 `Convert_int8`；`int8_tensor.ckpt` 当前未引用；
  `add_result.ckpt` 实际用于 LayerNorm 查表加法，Softmax 当前仅加载但 forward
  未使用其查表求和路径。
- 将 `lookup_tables/*.ckpt` 加入 `.gitignore`，避免误提交大体积运行资产。

## 2026-06-06 16:39:40 +0800

- 完成查找表本地化验证：三份副本与源文件逐字节一致，目录总大小约 3.8 GB。
- 完成相关模块语法检查和 `git diff --check`。
- 实际构造并运行 `Convert_int8`，确认能够从本地 `toint_tensor.ckpt` 加载并完成转换。
- 确认 `diffusion_policy/module/` 已不再引用原 `/datadisk` 或外部
  `add_result.ckpt` 路径。

## 2026-06-06 18:14:00 +0800

- 重新核对 `real_demo/umi_for_train` 三份查找表在训练主路径中的实际使用情况。
- `int8_tensor.ckpt` 在 `umi_for_train` 中没有代码引用，完全未使用。
- `add_result.ckpt` 会被 Softmax 构造函数加载，但 Softmax forward 使用
  `torch.sum`，查表加法树被注释，因此不参与训练数值计算；action transformer
  同时使用 `nn.LayerNorm`，不会实例化自定义查表 LayerNorm。
- `toint_tensor.ckpt` 会被 `Convert_int8` 加载，并被 quant Linear A/B 的
  forward 实际调用。量化 Linear 使用自定义 `autograd.Function` 手写 backward，
  因此训练反传不经过不可导的查表索引操作。
- 结论：严格复现 `umi_for_train` 的 action QAT 时，应保留
  `toint_tensor.ckpt`，不应让 Softmax/LayerNorm 使用 `add_result.ckpt`
  bit-true 加法路径。

## 2026-06-06 20:56:55 +0800

- 按用户确认的训练数值路径回退此前融合的 demo bit-true 逻辑。
- action QAT Decoder 的三个 norm 和 final norm 全部恢复为无 affine 的
  `nn.LayerNorm`，与 `real_demo/umi_for_train` action 主路径一致。
- Softmax 删除 `add_result.ckpt` 加载和未启用的查表加法树，保留分段指数近似
  与 `torch.sum` 分母计算。
- 删除 `layernorm_768.py`；`module/layernorm.py` 删除查表逻辑，保留普通可反传
  FP16 实现供非 action module 入口兼容使用。
- 保留 QAT FFN `4*n_emb` 规则：256 为 1024，768 为 3072。
- 确认运行代码中唯一 `.ckpt` 加载点为 `toint_tensor.ckpt`；
  `int8_tensor.ckpt` 和 `add_result.ckpt` 保留但不加载。
- 256/768 QAT 结构验证、Softmax/LayerNorm 局部反传、单层完整 action QAT
  前向和反向 smoke test 均通过。

## 2026-06-06 21:26:53 +0800

- 核对 768 QAT 与 256 FP32 配置：QAT 为 768/8 heads/7 layers/FFN 3072，
  FP32 our 为 256/4 heads/8 layers/FFN 1024；两者视觉编码器保持可训练。
- 新增 `train_diffusion_transformer_umi_workspace_qat.yaml` 和
  `qat_checkpoint_util.py`。768 QAT 从 epoch 130 FP32 checkpoint 初始化，
  不恢复旧 epoch、optimizer 或量化 scale。
- 完整权重迁移校验通过：model 和 EMA 各迁移 382 个 QAT 目标参数；attention
  `in_proj_weight` 拆分为 Q/K/V，FP32 bias 和 LayerNorm affine 按设计忽略。
- 数值主路径与 `real_demo/umi_for_train` 核对完成：action transformer 仅有
  decoder FFN 从其硬编码 1024 改为 `4*n_emb` 的有效数值差异；Softmax 删除的
  是未启用 bit-true 代码，Quant Linear 差异仅为路径本地化和格式。
- GPU1 768 QAT 单步 smoke test通过：batch 1 loss 2.2811288834，batch 8
  loss 1.8587532043。默认 batch 32 会触发量化路径 CUDA device-side assert，
  因此正式配置改为 batch 8。
- GPU2 256 FP32 单步 smoke test通过：batch 1 loss 1.0142469406，batch 32
  loss 1.0675221682。默认 batch 64 OOM，因此正式配置改为 batch 32。
- 已使用 WandB online 启动长期训练：
  `umi_qat768_gpu1` 使用 GPU1，run `mdxqzt63`；
  `umi_fp32_256_gpu2` 使用 GPU2，run `eulzbr0o`。

## 2026-06-06 21:45:00 +0800

- 按用户要求，将独立 QAT YAML 的 train/val batch 从 8 调整为 16；保留独立
  QAT YAML 和标准 FP32 YAML，不删除配置。
- 确认训练命令明确使用
  `/home/liyixuan23/miniforge3/envs/umi/bin/python`，即 umi 环境。
- GPU3 上 batch 16 连续完成 5-step QAT smoke test。
- 使用外部 hook 诊断查表越界，不修改 `module/` 芯片算子逻辑。
- 确认 `toint_tensor.ckpt` 长度为 45090；所有 `[-130, 130]` 有限值以及
  正负 Inf 经 clamp 后索引合法，FP16 NaN 会映射为越界索引 55312。
- 定位首个非有限输出来自 decoder cross-attention 的
  `quant_linear_b.fc_out`。现场输入、weight、quant_weight 和 input_delta
  均有限，但部分 scaling factor 转 FP16 时超过 65504 变成 Inf，随后
  `0 * Inf` 产生 NaN，最终导致 `Convert_int8` 查表索引越界。
- 结论：查表越界是 scaling factor FP16 溢出的后果，不应通过修改查表索引
  掩盖；可在芯片算子外评估梯度裁剪或降低 action transformer 学习率。

## 2026-06-06 22:40:10 +0800

- 按用户要求仅调整 QAT 训练超参数，不修改 `module/` 芯片算子：action
  transformer LR 从 `3e-4` 降为 `1e-4`，仍高于 ViT/obs encoder 的 `3e-5`；
  train/val batch 使用 32，未加入梯度裁剪。
- batch 32、action LR `1e-4` 的外部有限值诊断连续完成 101 个 step，未出现
  NaN、scaling factor 溢出或查表越界；GPU3 占用约 31.9 GiB。
- 正式 workspace 首次启动仍在第 1 次后续前向触发查表越界。对齐实验确认
  `num_workers=8` 不是原因；加入正式配置的 2000-step warmup 可稳定复现：
  第 0 步 LR 为 0，下一次前向在
  `decoder.layers.0.multihead_attn.fc_out` 产生非有限 scaling factor。
- QAT 独立配置新增 `training.lr_warmup_steps=0`。关闭 warmup 后正式 GPU1
  QAT 已正常越过原先立即失败点，启动参数为 batch 32、action LR `1e-4`、
  ViT LR `3e-5`；W&B run 为 `07fsc6fz`。
- GPU2 的 256 FP32 长期训练保持运行。

## 2026-06-06 22:44:13 +0800

- 直接核对关闭 warmup 后的正式 GPU1 QAT：已运行到 `global_step=43`，越过此前
  第 36 步失败点；当前 loss 有限，最新值约 `0.527`，tmux 会话仍在运行。
- GPU1 QAT 当前占用约 35.9 GiB，batch 32；GPU2 FP32 已运行到
  `global_step=3341`，两项正式训练均正常。

## 2026-06-06 22:47:49 +0800

- 修正上一条状态：GPU1 正式 QAT 随后在第 46 次前向再次触发
  `Convert_int8` 查表越界并退出；降低 LR 和关闭 warmup 延迟了溢出，但尚未
  解决长期稳定性，因此当前 QAT 配置不能认为可以完整训练。
- 数据集在 batch 32 下每个 epoch 为 2303 step。按实测速度，768 QAT 约
  6.1 秒/step，即约 3 小时 54 分钟/epoch；256 FP32 约 1.39 秒/step，即约
  53 分钟/epoch。GPU2 FP32 仍正常运行。

## 2026-06-07 09:05:00 +0800

- 建立 `.codex/qat_stability_experiment.py` 精确算子诊断，确认 batch32 在没有
  backward 和参数更新时也会因 pretrained 映射后的数值范围失败；降低 LR 和
  梯度裁剪不是根因修复。
- 验证 pretrained 精确 QAT 的 `batch=1 + gradient accumulation=32`：已超过
  140 个样本、完成多次 optimizer 更新且未发生 scaling factor 溢出，近期
  loss 可到约 0.2-0.5；但自定义 Softmax 仍会偶发极端负值及 row-sum 异常，
  batch1 不能解决 Softmax 尾部问题。
- 发现量化层 train/eval scale 语义：train 模式每个 batch 重算 scale；eval
  模式复用 checkpoint 保存的固定 scale。成功 256 QAT checkpoint 保存了 130
  个 input/output delta，部署脚本直接调用 `policy.eval()`，没有独立校准。
- pretrained 映射模型的固定-scale eval 在第 8 个样本产生非有限值；随机初始
  模型固定-scale eval 连续完成 300 个样本，最大 raw scale 约 343，平均
  Convert 饱和率约 0.005%，未溢出。说明 from-scratch 很可能规避 pretrained
  权重范围失配，但最终仍必须验证固定-scale 部署路径。
- 修复 workspace 梯度累积边界：此前 global_step=0 时会在第一个 microbatch
  立即 optimizer.step；现在累计满 N 个 microbatch 后更新，EMA 也仅在 optimizer
  更新时推进。
- 区分训练超参数：pretrained QAT 候选使用 action LR 1e-4、无 warmup；
  from-scratch 必须按标准配置使用 action LR 3e-4、2000 optimizer-step warmup。
  早期同 LR、无 warmup 的 scratch 实验只作为溢出压力测试，不用于收敛判断。
- 已启动标准 scratch 精确 QAT 对照：batch32 与 batch1+accumulate32 均使用
  LR 3e-4 + warmup2000。初期 scale 约 286-334，Softmax 为正，未出现溢出。

## 2026-06-07 09:35:00 +0800

- 完成统一短窗口“训练后立即固定-scale eval”对照，三组均使用完整精确芯片
  算子，不修改 `module/`：
  - pretrained→QAT，batch1+accumulate32，LR 1e-4/no warmup：训练 64 个
    样本无非有限值，但 Softmax 最小值达到 -47.47；固定-scale eval 100 个
    样本中最小值达到 -244，Convert 输入最大 9400。batch1 仅延缓失败，未解决
    attention/Softmax 数值错误。
  - scratch，batch1+accumulate32，LR 3e-4/warmup2000：训练和固定-scale
    eval 各通过 64/100 个样本；Softmax 始终为正，最大 raw scale 365/314。
  - scratch，batch32，LR 3e-4/warmup2000：训练 loss 前10步均值 1.106、
    末10步均值 0.690；固定-scale eval 100 步 loss 约 0.54，Softmax 始终
    为正，最大 raw scale 351/305，平均 Convert 饱和率约 0.00041%。
- 当前证据支持：from-scratch 可规避 pretrained checkpoint 映射造成的初始
  范围失配；为避免溢出并不要求 batch1，标准 batch32 是当前更好的候选。
  该结论仍是早期验证，需进行长步数训练并持续检查 Softmax 非负、row sum、
  raw scale、Convert 饱和率和固定-scale eval。

## 2026-06-07 09:45:00 +0800

- 定位极端负 Softmax 的代码根因。自定义实现的分段线性拟合不保证非负：
  `[-4,-8]` 段约在 -7.24 后变负，`[-8,-16]` 段约在 -13.77 后变负。
- 更严重的是实现构造了 16 段、最后一段 `[-32,-inf)` 输出零，但 forward
  仅循环 `range(15)`；小于等于 -32 的 shifted logits 保留为原始负数。
  最小复现 `[0,-100]` 输出约 `[-0.0101,1.0098]`。
- 确认 `real_demo/umi_for_train` action 量化主路径使用同一自定义 Softmax 和
  同一 `range(15)` 主计算；其中 add_result/sum_tree 代码存在但 forward 未用。
- 通过外部诊断 hook 临时替换为 `torch.softmax`，不修改 `module/`：
  pretrained batch32 早期纯前向和短训练均保持非负、行和误差约 4e-7，未复现
  立即溢出。该替换可能解决失败链，但不符合“严格复现芯片算子”的最终要求；
  第16段已构造却未使用强烈提示软件模型 bug，需芯片负责人确认真实尾区行为。

## 2026-06-07 09:41:22 +0800

- 审计并移除三个与 QAT 主任务无关的遗留改动：
  `scripts/eval_real_robot.py`、`scripts/replay_real_robot.py` 中仅有
  `cuda:0` 到 `cuda` 和空白格式变化；`diffusion_policy/schedule/ddim_schedule.py`
  中仅修改了一行注释。三个文件现均恢复至 Git 基线。
- 通过 Hydra compose 确认标准 FP32 与 QAT 配置实际都实例化
  `diffusers.DDIMScheduler`；仓库自定义
  `diffusion_policy.schedule.ddim_schedule.DDIMScheduler` 当前未被这两条路径使用。

## 2026-06-07 09:50:00 +0800

- 为七组多卡对照实验新增 `.codex/multigpu_experiments.md`，所有命令通过
  `BATCH_SIZE_PER_GPU` 配置每卡 batch，并明确全局 batch 的计算方式。
- 新增 256 维 from-scratch QAT 配置
  `train_diffusion_transformer_umi_workspace_qat_our.yaml`，继承 our/256 架构并仅
  将 policy 主入口切换到芯片同构 QAT policy。
- 新增 `scripts/run_pick_night_multigpu.sh`，用于在同一 GPU 集合上串行执行
  `example_pick_night` 的 FP32 our/256、QAT our/256、QAT 768 三项实验。
- 新数据集上的 QAT 实验统一按 from-scratch 启动；768 QAT 显式清空旧数据集
  checkpoint，并恢复 LR 3e-4 和 2000-step warmup。

## 2026-06-07 09:56:00 +0800

- Hydra 解析验证通过：FP32 768、FP32 our/256、QAT 768 scratch、QAT
  our/256 均正确解析模型入口、维度、每卡 batch、LR 和 warmup；脚本通过
  `bash -n`，相关 Python 文件通过 `compileall`。
- 提交 `a3e7b15` 已推送到远端 `iivy_umi/qat`。
- 已启动 tmux 会话 `pick_night_multigpu`，在 GPU0/1/3 上以每卡 batch 16、
  全局 batch 48 串行执行三项 `example_pick_night` 实验。第一项 FP32
  our/256 已超过 100 step，三卡各占约 15.3 GiB，训练正常；完成后会自动启动
  QAT our/256 和 QAT 768 scratch。
- 保留 GPU2 上原有 `umi_fp32_256_gpu2` 训练，不与新任务共享 GPU。
