#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


METRIC_KEYS = (
    "train_loss",
    "val_loss",
    "train_action_mse_error",
    "train_action_mse_error_raw_model",
    "val_action_mse_error",
    "val_action_mse_error_raw_model",
    "lr",
)


def as_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_logs(path):
    rows = []
    with Path(path).open("r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def epoch_rows(rows):
    result = []
    seen_epochs = set()
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        has_epoch_metric = any(key in row for key in METRIC_KEYS[1:])
        if has_epoch_metric or epoch not in seen_epochs:
            result.append(row)
            seen_epochs.add(epoch)
    return result


def trend(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return "n/a"
    first = values[0]
    last = values[-1]
    delta = last - first
    direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
    return f"{direction} ({first:.6g} -> {last:.6g}, delta={delta:.6g})"


def first_sustained_increase(rows, key, patience):
    best = None
    streak = 0
    for row in rows:
        value = as_float(row.get(key))
        if value is None:
            continue
        if best is None or value <= best:
            best = value
            streak = 0
            continue
        streak += 1
        if streak >= patience:
            return row.get("epoch"), value, best
    return None


def print_table(rows, keys):
    header = ["epoch", "global_step"] + list(keys)
    print("\t".join(header))
    for row in rows:
        cells = [str(row.get("epoch", "")), str(row.get("global_step", ""))]
        for key in keys:
            value = as_float(row.get(key))
            cells.append("" if value is None else f"{value:.8g}")
        print("\t".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", help="Path to logs.json.txt")
    parser.add_argument("--last", type=int, default=30, help="Number of epoch/sample rows to print")
    parser.add_argument("--patience", type=int, default=3, help="Consecutive worse points before flagging increase")
    args = parser.parse_args()

    rows = load_logs(args.logs)
    samples = epoch_rows(rows)
    keys = [key for key in METRIC_KEYS if any(key in row for row in samples)]

    print(f"loaded_rows={len(rows)} sample_or_epoch_rows={len(samples)} path={args.logs}")
    if not samples:
        return

    print_table(samples[-args.last:], keys)
    print()
    for key in keys:
        values = [as_float(row.get(key)) for row in samples]
        print(f"{key}: {trend(values)}")
        flag = first_sustained_increase(samples, key, args.patience)
        if flag is not None:
            epoch, value, best = flag
            print(f"  first_sustained_increase: epoch={epoch} value={value:.6g} best_before={best:.6g}")


if __name__ == "__main__":
    main()
