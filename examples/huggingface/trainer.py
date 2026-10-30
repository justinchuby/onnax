# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train existing ONNX LoRA using datasets and Trainer."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from examples.huggingface.bridge import OnnxModule
from examples.lora.model import adapt, base_model
from examples.lora.train import problem, reference, session


class RegressionModel(torch.nn.Module):
    def __init__(self, *, nf4=False):
        super().__init__()
        w, c, _, _, _, _, params = problem()
        graph = adapt(base_model(w, c))
        if nf4:
            from examples.lora.nf4 import lower

            graph, _ = lower(graph)
        self.onnx = OnnxModule(graph, params)

    def forward(self, features, labels=None):
        prediction = self.onnx(features=features)
        loss = None if labels is None else (prediction - labels).square().mean()
        return {"loss": loss, "logits": prediction}


class StopAt(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = control.should_training_stop = True


def make_trainer(directory, *, stop=None, nf4=False):
    _, _, x, y, vx, vy, _ = problem()
    args = TrainingArguments(
        output_dir=directory,
        use_cpu=True,
        max_steps=200,
        learning_rate=0.03,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        weight_decay=0,
        max_grad_norm=0,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        save_steps=100,
        save_strategy="steps",
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=20261029,
    )
    return Trainer(
        model=RegressionModel(nf4=nf4),
        args=args,
        train_dataset=Dataset.from_dict({"features": x, "labels": y}),
        eval_dataset=Dataset.from_dict({"features": vx, "labels": vy}),
        callbacks=[] if stop is None else [StopAt(stop)],
    )


def worker(directory, *, resume=None, stop=None, nf4=False):
    torch.set_num_threads(1)
    trainer = make_trainer(directory, stop=stop, nf4=nf4)
    trainer.train(resume_from_checkpoint=resume)
    trainer.model.onnx.export(os.path.join(directory, "export"))
    return trainer


def run(directory, *, nf4=False):
    torch.set_num_threads(1)
    full = worker(os.path.join(directory, "continuous"), nf4=nf4)
    interrupted = os.path.join(directory, "resumed")
    worker(interrupted, stop=100, nf4=nf4)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.trainer",
            "--worker",
            "--output-dir",
            interrupted,
            "--resume",
            os.path.join(interrupted, "checkpoint-100"),
            *(["--nf4"] if nf4 else []),
        ],
        check=True,
    )
    with np.load(os.path.join(interrupted, "export", "adapter.npz")) as saved:
        for name, array in full.model.onnx.arrays().items():
            np.testing.assert_array_equal(array, saved[name])
    w, c, x, y, vx, vy, initial = problem()
    if nf4:
        from examples.lora.nf4 import dequantize, quantize

        w = dequantize(*quantize(w), w.shape)
        c = dequantize(*quantize(c), c.shape)
    final = full.model.onnx.arrays()
    report = {
        "steps": 200,
        "checkpoint_replay": True,
        "train_loss": [float(reference(x, y, p, w, c)[1]) for p in (initial, final)],
        "validation_loss": [float(reference(vx, vy, p, w, c)[1]) for p in (initial, final)],
    }
    assert report["train_loss"][1] < report["train_loss"][0] * 0.5
    assert report["validation_loss"][1] < report["validation_loss"][0] * 0.5
    prediction = session(os.path.join(directory, "continuous", "export", "forward.onnx")).run(
        None, {"features": vx}
    )[0]
    np.testing.assert_allclose(
        prediction, reference(vx, vy, final, w, c)[0], rtol=1e-4, atol=1e-6
    )
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="trainer_models")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--nf4", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.output_dir, resume=args.resume, nf4=args.nf4)
    else:
        print(json.dumps(run(args.output_dir, nf4=args.nf4), indent=2))
