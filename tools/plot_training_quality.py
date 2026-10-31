# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Plot paired FP32/BF16 quality reports without importing the training runtime."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"torch_cpu": "PyTorch CPU", "mlx": "ONNX / MLX"}
COLORS = {"torch_cpu": "#0072B2", "mlx": "#D55E00"}
MATCHED = (
    "model",
    "revision",
    "nf4",
    "corpus_sha256",
    "train_lines",
    "heldout_lines",
    "steps",
    "microbatch",
    "accumulation",
    "optimizer",
)


def plot(fp32_path, bf16_path, output, *, preview=None):
    with open(fp32_path) as stream:
        fp32 = json.load(stream)
    with open(bf16_path) as stream:
        bf16 = json.load(stream)
    if any(fp32["settings"][key] != bf16["settings"][key] for key in MATCHED):
        raise ValueError("Select matched precision experiments")
    if fp32["seeds"] != bf16["seeds"]:
        raise ValueError("Trial seeds differ")
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "onnax-quality"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for report, precision, ax in zip((fp32, bf16), ("fp32", "bf16"), axes, strict=True):
        if report["settings"]["projection_precision"] != precision:
            raise ValueError("Report precision differs from the selected panel")
        for component in report["components"]:
            engine = component["engines"][0]
            if engine not in LABELS:
                raise ValueError(f"Unexpected engine: {engine}")
            trials = sorted(component["trials"], key=lambda row: row["seed"])
            if [row["seed"] for row in trials] != sorted(report["seeds"]):
                raise ValueError("Incomplete engine trials")
            steps = [row["step"] for row in trials[0]["validation_curve"]]
            if any([p["step"] for p in row["validation_curve"]] != steps for row in trials):
                raise ValueError("Validation checkpoints differ")
            values = np.asarray(
                [[p["loss"] for p in row["validation_curve"]] for row in trials]
            )
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite validation curve")
            mean = values.mean(axis=0)
            deviation = values.std(axis=0, ddof=1) if len(trials) > 1 else np.zeros_like(mean)
            ax.fill_between(
                steps,
                mean - deviation,
                mean + deviation,
                color=COLORS[engine],
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                steps,
                mean,
                color=COLORS[engine],
                label=LABELS[engine],
                linestyle="-" if engine == "torch_cpu" else "--",
                marker="o" if engine == "torch_cpu" else "s",
                markersize=5,
                markerfacecolor=COLORS[engine] if engine == "torch_cpu" else "white",
            )
        ax.set_title(f"{precision.upper()} projections", fontsize=11)
        ax.set_xlabel("Optimizer updates")
        ax.set_xticks(steps)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Held-out cross-entropy (nats/token)")
    settings = fp32["settings"]
    fig.suptitle(
        f"{settings['model']}: {settings['train_lines']} training / "
        f"{settings['heldout_lines']} held-out sequences, {len(fp32['seeds'])} seeds\n"
        "Lines: seed means; bands: seed standard deviations",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(
        output, metadata={"Date": None, "Creator": f"Matplotlib {matplotlib.__version__}"}
    )
    if preview:
        fig.savefig(preview, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fp32")
    parser.add_argument("bf16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview")
    args = parser.parse_args()
    plot(args.fp32, args.bf16, args.output, preview=args.preview)


if __name__ == "__main__":
    main()
