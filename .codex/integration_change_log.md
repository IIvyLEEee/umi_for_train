# QAT 整合逐文件变更记录

## 2026-06-06 训练启动补充

- 新增 768 QAT 独立配置，设置 QAT policy、FP32 初始化 checkpoint 和已验证的
  batch 8。
- 新增 FP32 到 QAT checkpoint 映射工具，拆分 attention Q/K/V，迁移 model
  和 EMA，并严格检查全部 QAT 可训练参数都被初始化。
- Transformer Timm workspace 支持 `training.init_checkpoint`，与 resume 互斥；
  rollout、sample、checkpoint 周期允许负数用于 smoke test 禁用。
- 256 FP32 `our` 配置 batch 从 64 调整为已验证可运行的 32。

## 新增文件

- `diffusion_policy/policy/diffusion_transformer_timm_qat_policy.py`
  - 从当前 FP32 policy 派生独立入口。
  - 唯一模型入口差异是导入 `module/transformer_for_action_diffusion.py`。
  - 保留当前 policy 的训练、损失、scheduler 和推理噪声行为。
- `diffusion_policy/config/train_diffusion_transformer_umi_workspace_our.yaml`
  - 引入 `umi_for_train` 的 our 配置：256 维、4 heads、8 layers。
  - 将训练设备改为逻辑 `cuda:0`。
  - 将 `training.resume` 和 `logging.resume` 改为 `False`。
## QAT 算子链

- `module/transformer_for_action_diffusion.py`
  - 以 `umi_for_train` 为基准同步。
  - input embedding 和 output head 从普通自定义 Linear 改为 `quant_linear_b`。
  - final LayerNorm 使用无 affine 的 `nn.LayerNorm`，与 train 主路径一致。
- `module/decoder.py`
  - 以 `umi_for_train` 为基准同步 attention 和 FFN 数据流。
  - FFN 改为 `quant_linear_a -> relu -> quant_linear_c`。
  - 将参考实现固定的 1024 改为构造参数 `dim_feedforward`：
    256 保持 1024，768 使用 3072。
  - 三个 norm 保持无 affine 的 `nn.LayerNorm`。
- `module/multihead_attn.py`
  - 同步 `umi_for_train`：Q/K/V 使用 `quant_linear_a`，输出使用
    `quant_linear_b`，attention 使用自定义 Softmax。
- `module/quant_linear_a.py`
  - 同步 `umi_for_train` 的 FP16 输入、INT8 输出和 `output_delta` 传递行为。
- `module/quant_linear_b.py`
  - 同步 `umi_for_train` 的 FP16 输入、FP16 输出量化行为。
- `module/quant_linear_c.py`
  - 同步 `umi_for_train` 的已量化输入、FP16 输出行为。
- `module/softmax.py`
  - 同步 `umi_for_train` 的实际训练数值路径：分段指数近似加 `torch.sum`。
  - 删除无效的 `add_result.ckpt` 加载和 bit-true 加法树代码。
- `module/tofp16.py`、`module/toint8.py`
  - 同步 `umi_for_train` 的芯片转换与量化尺度实现。
- `module/layernorm.py`
  - 删除 demo bit-true 查表加法逻辑，保留可反传的普通 FP16 LayerNorm，
    供非 action module 入口兼容使用。

## 设备适配

- `quant_linear_a.py`、`quant_linear_b.py`、`quant_linear_c.py`
  - 运行张量跟随 `input.device`；示例代码不再绑定 `cuda:2`。
- `softmax.py`
  - 示例代码不再绑定 `cuda:2`。
- `scripts/eval_real_robot.py`、`scripts/replay_real_robot.py`
  - 设备从 `cuda:0` 改为逻辑 `cuda`。
- `diffusion_policy/schedule/ddim_schedule.py`
  - 注释示例中的 `cuda:0` 改为逻辑 `cuda`。

## 本地查找表

- `lookup_tables/`
  - 从 `eval_for_demo_0628` 复制 `int8_tensor.ckpt` 和 `toint_tensor.ckpt`。
  - 从 demo LayerNorm 实际引用路径复制 `add_result.ckpt`；源项目顶层不存在该文件。
- `module/lookup_table.py`
  - 新增基于仓库位置的统一查找表路径解析和缺失文件报错。
- `module/toint8.py`
  - `Convert_int8` 改为加载本地 `lookup_tables/toint_tensor.ckpt`。
- `module/softmax.py`、`module/layernorm.py`
  - 不加载 `add_result.ckpt`。
- `.gitignore`
  - 忽略 `lookup_tables/*.ckpt`，避免误提交 3.8 GB 查找表资产。

## 未改动的行为

- 现有 FP32 policy 和标准 768 YAML 保持不变。
- `int8_tensor.ckpt` 和 `add_result.ckpt` 保留在本地但不加载。
- 未引入 demo 的 `quant_selfattn.py`、`quant_multiheadattn.py` 或调试退出逻辑。
- 未引入 `umi_for_train` 的固定种子自定义推理噪声生成器。
