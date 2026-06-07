import dill
import torch


def _copy_fp32_policy_to_qat(qat_policy, fp32_state_dict):
    qat_state = qat_policy.state_dict()
    copied = set()
    consumed_source = set()

    for key, value in fp32_state_dict.items():
        if key in qat_state and qat_state[key].shape == value.shape:
            qat_state[key] = value
            copied.add(key)
            consumed_source.add(key)

    for layer_idx in range(qat_policy.model.decoder.num_layers):
        for attention_name in ("self_attn", "multihead_attn"):
            fp32_prefix = (
                f"model.decoder.layers.{layer_idx}.{attention_name}"
            )
            qat_prefix = fp32_prefix
            in_proj_key = f"{fp32_prefix}.in_proj_weight"
            in_proj_weight = fp32_state_dict[in_proj_key]
            q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
            consumed_source.add(in_proj_key)
            consumed_source.add(f"{fp32_prefix}.out_proj.weight")

            attention_mapping = {
                f"{qat_prefix}.queries.weight": q_weight,
                f"{qat_prefix}.keys.weight": k_weight,
                f"{qat_prefix}.values.weight": v_weight,
                f"{qat_prefix}.fc_out.weight": fp32_state_dict[
                    f"{fp32_prefix}.out_proj.weight"
                ],
            }
            for target_key, value in attention_mapping.items():
                if qat_state[target_key].shape != value.shape:
                    raise RuntimeError(
                        f"QAT checkpoint shape mismatch for {target_key}: "
                        f"target={tuple(qat_state[target_key].shape)}, "
                        f"source={tuple(value.shape)}"
                    )
                qat_state[target_key] = value
                copied.add(target_key)

    qat_policy.load_state_dict(qat_state, strict=True)

    trainable_target_keys = {
        key for key, value in qat_policy.named_parameters() if value.requires_grad
    }
    missing_trainable = sorted(trainable_target_keys - copied)
    if missing_trainable:
        raise RuntimeError(
            "FP32 checkpoint did not initialize trainable QAT parameters: "
            + ", ".join(missing_trainable)
        )

    ignored_source = sorted(set(fp32_state_dict) - consumed_source)
    return {
        "copied": len(copied),
        "ignored_source": ignored_source,
    }


def initialize_qat_from_fp32_checkpoint(model, ema_model, checkpoint_path):
    with open(checkpoint_path, "rb") as checkpoint_file:
        payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            pickle_module=dill,
        )

    state_dicts = payload["state_dicts"]
    model_report = _copy_fp32_policy_to_qat(model, state_dicts["model"])
    ema_report = None
    if ema_model is not None:
        source_key = "ema_model" if "ema_model" in state_dicts else "model"
        ema_report = _copy_fp32_policy_to_qat(ema_model, state_dicts[source_key])

    return {
        "model": model_report,
        "ema_model": ema_report,
    }
