import argparse
import json
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.qat_checkpoint_util import initialize_qat_from_fp32_checkpoint
from diffusion_policy.module import quant_linear_a, quant_linear_b, quant_linear_c
from diffusion_policy.module.softmax import Softmax


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", default="/tmp/qat_decomposition_probe.jsonl")
    return parser.parse_args()


def torch_softmax(self, input_tensor):
    return torch.softmax(input_tensor.float(), dim=-1)


@contextmanager
def action_mode(
    model,
    source_state,
    *,
    float_linear,
    standard_softmax,
    source_bias=False,
    source_ln_affine=False,
    gelu=False,
):
    originals = []
    activations = []
    for name, module in model.named_modules():
        replacement = None
        bias = source_linear_bias(name, source_state) if source_bias else None
        if float_linear and isinstance(module, quant_linear_a.Linear):
            def float_a(
                self, input_tensor, parameter, dequant_input, *, current_bias=bias
            ):
                output = F.linear(
                    dequant_input.float() * parameter,
                    self.weight.float(),
                    current_bias,
                )
                return output, output.new_tensor(1.0)

            replacement = float_a
        elif float_linear and isinstance(module, quant_linear_b.Linear):
            def float_b(
                self,
                input_tensor,
                parameter,
                dequant_input=None,
                out_parameter=None,
                *,
                current_bias=bias,
            ):
                if dequant_input is None:
                    dequant_input = input_tensor
                output = F.linear(
                    dequant_input.float() * parameter,
                    self.weight.float(),
                    current_bias,
                )
                if out_parameter is not None:
                    output = output / out_parameter
                return output

            replacement = float_b
        elif float_linear and isinstance(module, quant_linear_c.Linear):
            def float_c(self, input_tensor, input_delta, *, current_bias=bias):
                return F.linear(
                    input_tensor.float() * input_delta,
                    self.weight.float(),
                    current_bias,
                )

            replacement = float_c
        elif standard_softmax and isinstance(module, Softmax):
            replacement = torch_softmax
        elif source_ln_affine and isinstance(module, torch.nn.LayerNorm):
            weight = source_state[f"{name}.weight"]
            bias = source_state[f"{name}.bias"]

            def affine_layer_norm(
                self, input_tensor, *, current_weight=weight, current_bias=bias
            ):
                return F.layer_norm(
                    input_tensor,
                    self.normalized_shape,
                    current_weight,
                    current_bias,
                    self.eps,
                )

            replacement = affine_layer_norm
        if replacement is not None:
            originals.append((module, module.forward))
            module.forward = types.MethodType(replacement, module)
        if gelu and hasattr(module, "activation"):
            activations.append((module, module.activation))
            module.activation = F.gelu
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original
        for module, original in activations:
            module.activation = original


def source_linear_bias(name, source_state):
    direct_key = f"{name}.bias"
    if direct_key in source_state:
        return source_state[direct_key]
    for attention_name in ("self_attn", "multihead_attn"):
        marker = f".{attention_name}."
        if marker not in name:
            continue
        prefix, projection = name.rsplit(".", 1)
        if projection == "fc_out":
            return source_state[f"{prefix}.out_proj.bias"]
        if projection in ("queries", "keys", "values"):
            bias = source_state[f"{prefix}.in_proj_bias"]
            index = ("queries", "keys", "values").index(projection)
            return bias.chunk(3)[index]
    return None


def register_layer_capture(model, storage):
    handles = []
    for name, module in model.named_modules():
        if (
            name.startswith("decoder.layers.")
            and name.count(".") == 2
        ):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, layer=name: storage.__setitem__(
                        layer, output.detach().float()
                    )
                )
            )
    return handles


def metrics(prediction, teacher_prediction, target, layers, teacher_layers):
    result = {
        "loss": float(F.mse_loss(prediction.float(), target.float())),
        "teacher_mse": float(
            F.mse_loss(prediction.float(), teacher_prediction.float())
        ),
        "prediction_absmax": float(prediction.abs().max()),
    }
    result["layer_mse"] = {
        name: float(F.mse_loss(output, teacher_layers[name]))
        for name, output in layers.items()
        if name in teacher_layers
    }
    return result


def main():
    args = parse_args()
    output_path = Path(args.output)

    with initialize(version_base=None, config_path="../diffusion_policy/config"):
        cfg = compose(
            config_name="train_diffusion_transformer_umi_workspace_qat",
            overrides=[
                f"dataloader.batch_size={args.batch_size}",
                "dataloader.num_workers=4",
                "dataloader.persistent_workers=true",
            ],
        )
        teacher_cfg = compose(
            config_name="train_diffusion_transformer_umi_workspace",
            overrides=[
                f"dataloader.batch_size={args.batch_size}",
                "dataloader.num_workers=4",
                "dataloader.persistent_workers=true",
            ],
        )

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = cfg.training.init_checkpoint
    with open(checkpoint, "rb") as checkpoint_file:
        payload = torch.load(checkpoint_file, map_location="cpu", pickle_module=dill)

    teacher = hydra.utils.instantiate(teacher_cfg.policy)
    teacher.load_state_dict(payload["state_dicts"]["model"], strict=True)
    student = hydra.utils.instantiate(cfg.policy)
    initialize_qat_from_fp32_checkpoint(student, None, checkpoint)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    teacher.set_normalizer(normalizer)
    student.set_normalizer(normalizer)
    dataloader = DataLoader(dataset, **cfg.dataloader)

    device = torch.device("cuda")
    teacher.to(device).eval()
    student.model.to(device).eval()
    source_state = {
        key.removeprefix("model."): value.to(device)
        for key, value in payload["state_dicts"]["model"].items()
        if key.startswith("model.")
    }
    modes = {
        "fp32_equivalent": {
            "float_linear": True,
            "standard_softmax": True,
            "source_bias": True,
            "source_ln_affine": True,
            "gelu": True,
        },
        "drop_bias": {
            "float_linear": True,
            "standard_softmax": True,
            "source_ln_affine": True,
            "gelu": True,
        },
        "drop_ln_affine": {
            "float_linear": True,
            "standard_softmax": True,
            "source_bias": True,
            "gelu": True,
        },
        "replace_gelu_with_relu": {
            "float_linear": True,
            "standard_softmax": True,
            "source_bias": True,
            "source_ln_affine": True,
        },
        "chip_float_torch_softmax": {
            "float_linear": True,
            "standard_softmax": True,
        },
        "chip_float_custom_softmax": {
            "float_linear": True,
            "standard_softmax": False,
        },
        "exact_qat_torch_softmax": {
            "float_linear": False,
            "standard_softmax": True,
        },
        "exact_qat_custom_softmax": {
            "float_linear": False,
            "standard_softmax": False,
        },
    }

    with output_path.open("w") as output_file:
        output_file.write(
            json.dumps(
                {
                    "event": "start",
                    "checkpoint": checkpoint,
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                }
            )
            + "\n"
        )

    for step, batch in enumerate(dataloader):
        if step >= args.steps:
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        nobs = teacher.normalizer.normalize(batch["obs"])
        trajectory = teacher.normalizer["action"].normalize(batch["action"])

        with torch.no_grad():
            cond = teacher.obs_encoder(nobs)
            noise = torch.randn_like(trajectory)
            noise_new = noise + teacher.input_pertub * torch.randn_like(trajectory)
            timesteps = torch.randint(
                0,
                teacher.noise_scheduler.config.num_train_timesteps,
                (trajectory.shape[0],),
                device=device,
            ).long()
            noisy = teacher.noise_scheduler.add_noise(
                trajectory, noise_new, timesteps
            )

            teacher_layers = {}
            handles = register_layer_capture(teacher.model, teacher_layers)
            teacher_prediction = teacher.model(noisy, timesteps, cond=cond)
            for handle in handles:
                handle.remove()

            record = {
                "event": "step",
                "step": step,
                "teacher_loss": float(
                    F.mse_loss(teacher_prediction.float(), noise.float())
                ),
                "teacher_prediction_absmax": float(teacher_prediction.abs().max()),
                "modes": {},
            }
            for mode_name, mode_options in modes.items():
                layers = {}
                handles = register_layer_capture(student.model, layers)
                try:
                    with action_mode(student.model, source_state, **mode_options):
                        prediction = student.model(noisy, timesteps, cond=cond)
                    record["modes"][mode_name] = metrics(
                        prediction,
                        teacher_prediction,
                        noise,
                        layers,
                        teacher_layers,
                    )
                except Exception as exc:
                    record["modes"][mode_name] = {"error": str(exc)}
                finally:
                    for handle in handles:
                        handle.remove()

        with output_path.open("a") as output_file:
            output_file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
