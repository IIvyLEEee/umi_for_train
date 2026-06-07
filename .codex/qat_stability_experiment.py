import argparse
import json
import random
import sys
import types
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import dill
from diffusers.optimization import get_scheduler
from einops import reduce
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.qat_checkpoint_util import initialize_qat_from_fp32_checkpoint
from diffusion_policy.module import quant_linear_a, quant_linear_b, quant_linear_c


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--post-eval-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-lr", type=float, default=1e-4)
    parser.add_argument("--obs-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--freeze-obs", action="store_true")
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--eval-model", action="store_true")
    parser.add_argument("--action-microbatch", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--grad-accumulate", type=int, default=1)
    parser.add_argument("--fc-out-input-divisor", type=float, default=None)
    parser.add_argument("--softmax-tail-floor", type=float, default=None)
    parser.add_argument("--torch-softmax", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--native-qat-checkpoint", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--n-emb", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/qat_stability")
    return parser.parse_args()


def tensor_status(tensor):
    finite = torch.isfinite(tensor)
    result = {
        "shape": list(tensor.shape),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
    }
    if finite.any():
        result["finite_min"] = float(tensor[finite].min())
        result["finite_max"] = float(tensor[finite].max())
    return result


def tensors_in(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item)


def raw_scaling_factor(module):
    if not hasattr(module, "input_delta") or not hasattr(module, "weight_delta"):
        return None
    scale = module.input_delta.float() * (2.0 ** (-module.weight_delta.float() + 1)) * 2**13
    if isinstance(module, quant_linear_a.Linear):
        scale = scale / module.output_delta.float()
    return scale


def compute_loss_action_microbatch(model, batch, microbatch_size):
    nobs = model.normalizer.normalize(batch["obs"])
    trajectory = model.normalizer["action"].normalize(batch["action"])
    obs_tokens = model.obs_encoder(nobs)
    noise = torch.randn(trajectory.shape, device=trajectory.device)
    noise_new = noise + model.input_pertub * torch.randn(
        trajectory.shape, device=trajectory.device
    )
    timesteps = torch.randint(
        0,
        model.noise_scheduler.config.num_train_timesteps,
        (trajectory.shape[0],),
        device=trajectory.device,
    ).long()
    noisy_trajectory = model.noise_scheduler.add_noise(
        trajectory, noise_new, timesteps
    )

    predictions = []
    for start in range(0, trajectory.shape[0], microbatch_size):
        end = start + microbatch_size
        predictions.append(
            model.model(
                noisy_trajectory[start:end],
                timesteps[start:end],
                cond=obs_tokens[start:end],
            )
        )
    pred = torch.cat(predictions, dim=0)
    target = noise
    loss = F.mse_loss(pred, target, reduction="none")
    loss = reduce(loss, "b ... -> b (...)", "mean")
    return loss.mean()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.name}.jsonl"

    with initialize(version_base=None, config_path="../diffusion_policy/config"):
        overrides = [
            f"dataloader.batch_size={args.batch_size}",
            "dataloader.num_workers=8",
            "dataloader.persistent_workers=true",
            f"policy.obs_encoder.pretrained={str(args.pretrained).lower()}",
            f"optimizer.lr={args.action_lr}",
            f"optimizer.obs_encoder_lr={args.obs_lr}",
        ]
        if args.n_emb is not None:
            overrides.extend([f"n_emb={args.n_emb}", f"policy.n_emb={args.n_emb}"])
        if args.n_head is not None:
            overrides.append(f"policy.n_head={args.n_head}")
        if args.n_layer is not None:
            overrides.append(f"policy.n_layer={args.n_layer}")
        cfg = compose(
            config_name="train_diffusion_transformer_umi_workspace_qat",
            overrides=overrides,
        )

    seed = int(cfg.training.seed) + args.seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = hydra.utils.instantiate(cfg.policy)
    init_checkpoint = args.init_checkpoint or cfg.training.init_checkpoint
    if args.scratch:
        pass
    elif args.native_qat_checkpoint:
        with open(init_checkpoint, "rb") as checkpoint_file:
            payload = torch.load(
                checkpoint_file, map_location="cpu", pickle_module=dill
            )
        model.load_state_dict(payload["state_dicts"]["model"], strict=False)
    else:
        initialize_qat_from_fp32_checkpoint(model, None, init_checkpoint)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    dataloader = DataLoader(dataset, **cfg.dataloader)

    state = {
        "step": -1,
        "first_nonfinite": None,
        "max_raw_scale": 0.0,
        "max_raw_scale_module": None,
        "softmax_min": float("inf"),
        "softmax_max_sum_error": 0.0,
        "convert_absmax": 0.0,
        "convert_saturated": 0,
        "convert_total": 0,
    }

    for module_name, module in model.named_modules():
        convert = getattr(module, "convert_int8", None)
        if convert is None:
            continue
        original_convert = convert.convert

        def checked_convert(x, *, name=module_name, original=original_convert):
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    "NONFINITE Convert_int8 input "
                    + json.dumps({"module": name, "tensor": tensor_status(x)})
                )
            state["convert_absmax"] = max(
                state["convert_absmax"], float(x.abs().max())
            )
            state["convert_saturated"] += int((x.abs() > 127.0).sum())
            state["convert_total"] += x.numel()
            return original(x)

        convert.convert = checked_convert

    if args.fc_out_input_divisor is not None:
        divisor = args.fc_out_input_divisor
        for module_name, module in model.named_modules():
            if not module_name.endswith((".self_attn.fc_out", ".multihead_attn.fc_out")):
                continue

            def divide_fc_out_inputs(_module, inputs, *, value=divisor):
                input_tensor, parameter, dequant_input, *remaining = inputs
                return (
                    input_tensor / value,
                    parameter,
                    dequant_input / value,
                    *remaining,
                )

            module.register_forward_pre_hook(divide_fc_out_inputs)

    if args.softmax_tail_floor is not None:
        tail_floor = args.softmax_tail_floor
        for module_name, module in model.named_modules():
            if not module_name.endswith(".softmax"):
                continue

            def floor_softmax_tail(_module, inputs, *, floor=tail_floor):
                input_tensor = inputs[0]
                input_max = input_tensor.max(dim=-1, keepdim=True).values
                return (torch.maximum(input_tensor, input_max + floor),)

            module.register_forward_pre_hook(floor_softmax_tail)

    if args.torch_softmax:
        for module_name, module in model.named_modules():
            if not module_name.endswith(".softmax"):
                continue

            def standard_softmax(_module, input_tensor):
                return torch.softmax(input_tensor.float(), dim=-1)

            module.forward = types.MethodType(standard_softmax, module)

    for module_name, module in model.named_modules():
        if module_name == "":
            continue

        def check_output(current_module, inputs, output, *, name=module_name):
            raw_scale = raw_scaling_factor(current_module)
            if raw_scale is not None:
                max_scale = float(raw_scale.max())
                if max_scale > state["max_raw_scale"]:
                    state["max_raw_scale"] = max_scale
                    state["max_raw_scale_module"] = name

            if name.endswith(".softmax"):
                for tensor in tensors_in(output):
                    state["softmax_min"] = min(
                        state["softmax_min"], float(tensor.min())
                    )
                    state["softmax_max_sum_error"] = max(
                        state["softmax_max_sum_error"],
                        float((tensor.sum(dim=-1) - 1).abs().max()),
                    )

            for tensor in tensors_in(output):
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    details = {
                        "module": name,
                        "class": current_module.__class__.__name__,
                        "output": tensor_status(tensor),
                    }
                    for attr_name in (
                        "weight",
                        "weight_delta",
                        "input_delta",
                        "output_delta",
                        "parameter",
                        "pre_parameter",
                        "scaling_factor",
                        "quant_weight",
                    ):
                        attr = getattr(current_module, attr_name, None)
                        if torch.is_tensor(attr):
                            details[attr_name] = tensor_status(attr)
                    raise RuntimeError("FIRST NONFINITE " + json.dumps(details))

        module.register_forward_hook(check_output)

    device = torch.device("cuda")
    model.to(device)
    optimizer = model.get_optimizer(**cfg.optimizer)
    num_training_steps = (
        len(dataloader) * int(cfg.training.num_epochs)
    ) // args.grad_accumulate
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    if args.eval_model:
        model.eval()
    elif args.freeze_obs:
        model.obs_encoder.eval()
        model.obs_encoder.requires_grad_(False)
    else:
        model.train()

    header = {
        "event": "start",
        "name": args.name,
        "steps": args.steps,
        "post_eval_steps": args.post_eval_steps,
        "seed": seed,
        "batch_size": args.batch_size,
        "action_lr": args.action_lr,
        "obs_lr": args.obs_lr,
        "warmup_steps": args.warmup_steps,
        "clip_grad": args.clip_grad,
        "freeze_obs": args.freeze_obs,
        "no_update": args.no_update,
        "pretrained": args.pretrained,
        "eval_model": args.eval_model,
        "action_microbatch": args.action_microbatch,
        "grad_accumulate": args.grad_accumulate,
        "fc_out_input_divisor": args.fc_out_input_divisor,
        "softmax_tail_floor": args.softmax_tail_floor,
        "torch_softmax": args.torch_softmax,
        "init_checkpoint": init_checkpoint,
        "native_qat_checkpoint": args.native_qat_checkpoint,
        "scratch": args.scratch,
        "n_emb": cfg.policy.n_emb,
        "n_head": cfg.policy.n_head,
        "n_layer": cfg.policy.n_layer,
    }
    output_path.write_text(json.dumps(header) + "\n")

    optimizer.zero_grad()
    try:
        for step, batch in enumerate(dataloader):
            if step >= args.steps:
                break
            state["step"] = step
            state["max_raw_scale"] = 0.0
            state["max_raw_scale_module"] = None
            state["softmax_min"] = float("inf")
            state["softmax_max_sum_error"] = 0.0
            state["convert_absmax"] = 0.0
            state["convert_saturated"] = 0
            state["convert_total"] = 0
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            if args.action_microbatch is not None and not args.no_update:
                raise ValueError(
                    "action microbatch is forward-only: current custom backward "
                    "stores mutable scale parameters"
                )
            if args.no_update:
                with torch.no_grad():
                    if args.action_microbatch is None:
                        loss = model(batch)
                    else:
                        loss = compute_loss_action_microbatch(
                            model, batch, args.action_microbatch
                        )
            else:
                loss = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NONFINITE loss: {float(loss)}")

            grad_norm = None
            if not args.no_update:
                (loss / args.grad_accumulate).backward()
                if (step + 1) % args.grad_accumulate == 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    )
                    if not np.isfinite(grad_norm):
                        raise RuntimeError(f"NONFINITE gradient norm: {grad_norm}")
                    if args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            record = {
                "event": "step",
                "step": step,
                "loss": float(loss),
                "grad_norm": grad_norm,
                "action_lr": optimizer.param_groups[0]["lr"],
                "max_raw_scale": state["max_raw_scale"],
                "max_raw_scale_module": state["max_raw_scale_module"],
                "fp16_scale_margin": 65504.0 - state["max_raw_scale"],
                "softmax_min": state["softmax_min"],
                "softmax_max_sum_error": state["softmax_max_sum_error"],
                "convert_absmax": state["convert_absmax"],
                "convert_saturation_rate": (
                    state["convert_saturated"] / state["convert_total"]
                    if state["convert_total"]
                    else 0.0
                ),
            }
            with output_path.open("a") as output_file:
                output_file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if args.post_eval_steps > 0:
            model.eval()
            with torch.no_grad():
                for eval_step, batch in enumerate(dataloader):
                    if eval_step >= args.post_eval_steps:
                        break
                    state["step"] = eval_step
                    state["max_raw_scale"] = 0.0
                    state["max_raw_scale_module"] = None
                    state["softmax_min"] = float("inf")
                    state["softmax_max_sum_error"] = 0.0
                    state["convert_absmax"] = 0.0
                    state["convert_saturated"] = 0
                    state["convert_total"] = 0
                    batch = dict_apply(
                        batch, lambda x: x.to(device, non_blocking=True)
                    )
                    loss = model(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"NONFINITE post-eval loss: {float(loss)}"
                        )
                    record = {
                        "event": "post_eval_step",
                        "step": eval_step,
                        "loss": float(loss),
                        "max_raw_scale": state["max_raw_scale"],
                        "max_raw_scale_module": state["max_raw_scale_module"],
                        "fp16_scale_margin": 65504.0 - state["max_raw_scale"],
                        "softmax_min": state["softmax_min"],
                        "softmax_max_sum_error": state["softmax_max_sum_error"],
                        "convert_absmax": state["convert_absmax"],
                        "convert_saturation_rate": (
                            state["convert_saturated"] / state["convert_total"]
                            if state["convert_total"]
                            else 0.0
                        ),
                    }
                    with output_path.open("a") as output_file:
                        output_file.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
    except Exception as exc:
        record = {
            "event": "failure",
            "step": state["step"],
            "error": str(exc),
            "max_raw_scale": state["max_raw_scale"],
            "max_raw_scale_module": state["max_raw_scale_module"],
        }
        with output_path.open("a") as output_file:
            output_file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        raise


if __name__ == "__main__":
    main()
