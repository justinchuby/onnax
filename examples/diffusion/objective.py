# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Export a scalar denoising objective for joint loss/gradient execution."""

import argparse
import json
import os
import tempfile

import onnx
import onnx_ir as ir
from onnxscript import optimizer

from onnax import FuseAttentionVJPPass, FuseBroadcastAdjointsPass
from onnax.torch import OnnxModule
from onnax.torch._serialization import file_digest


def prepare(
    training_run,
    destination,
    *,
    recompute_attention=False,
    projection_precision="fp32",
    session_factory=None,
):
    """Compose MSE with a qualified source; export independently loadable sessions."""
    if projection_precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Select fp32, bf16 or fp16 projection precision")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    from examples.diffusion.benchmark_unet import load_state

    _, interface, initial = load_state(training_run)
    for name, digest in interface["files"].items():
        if (
            os.path.basename(name) != name
            or file_digest(os.path.join(training_run, "export", name)) != digest
        ):
            raise ValueError(f"Source bundle integrity mismatch: {name!r}")
    source = ir.load(os.path.join(training_run, "export", "model.onnx"))
    if projection_precision != "fp32":
        from examples.huggingface.mixed_precision import precision_projections

        source = precision_projections(source, precision=projection_precision)
    names = ir.convenience.create_value_mapping(source.graph)
    if {"training.target", "training.loss"} & names.keys():
        raise ValueError("The source uses reserved objective names")
    if len(source.graph.outputs) != 1:
        raise ValueError("Select the UNet's single noise-prediction output")
    prediction = source.graph.outputs[0]
    target = ir.val(
        "training.target", dtype=prediction.dtype, shape=ir.Shape(prediction.shape)
    )
    source.graph.inputs.append(target)
    tape = ir.tape.Tape(source.graph)
    diff = tape.op("Sub", [prediction, target])
    loss = tape.op("ReduceMean", [tape.op("Mul", [diff, diff])], {"keepdims": 0})
    loss.name = "training.loss"
    source.graph.outputs.clear()
    source.graph.outputs.append(loss)

    def optimize(model):
        attention = FuseAttentionVJPPass(recompute=recompute_attention)(model).model
        return optimizer.optimize(FuseBroadcastAdjointsPass()(attention).model, inline=False)

    with OnnxModule(
        source,
        initial,
        input_gradients=[],
        eager_gradients=True,
        optimize=optimize,
        session_factory=session_factory,
    ) as module:
        module.export(destination)
        record = dict(
            version=1,
            base_identity=interface["identity"],
            base_model_sha256=interface["files"]["model.onnx"],
            objective_identity=module._identity,
            loss=loss.name,
            target=target.name,
            prediction=prediction.name,
            reduction="mean_squared_error",
            projection_precision=projection_precision,
        )
        with open(
            os.path.join(destination, "objective.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump(record, stream, indent=2)
    return record


def read_objective(directory, source_interface):
    with open(os.path.join(directory, "objective.json"), encoding="utf-8") as stream:
        record = json.load(stream)
    if (
        record.get("version") != 1
        or record["base_identity"] != source_interface["identity"]
        or record["base_model_sha256"] != source_interface["files"]["model.onnx"]
        or record["reduction"] != "mean_squared_error"
    ):
        raise ValueError("Objective belongs to a different source workload")
    source = ir.load(os.path.join(directory, "model.onnx"))
    policy = json.loads(source.metadata_props.get("onnax.example.precision", "null"))
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("Objective projection precision metadata must be an object")
    actual = (
        "fp32"
        if policy is None
        else {"bfloat16": "bf16", "float16": "fp16"}.get(policy.get("projection_compute"))
    )
    if actual is None or record.get("projection_precision", "fp32") != actual:
        raise ValueError("Objective projection precision differs from its source model")
    return record


def validate_checkpoint_objective(state, source_interface, objective=None):
    """Bind new checkpoints to their loss graph and retain legacy FP32 recovery."""
    expected = (
        source_interface["identity"] if objective is None else objective["objective_identity"]
    )
    if "objective_identity" in state:
        if state["objective_identity"] != expected:
            raise ValueError("Checkpoint belongs to a different training objective")
    elif objective is not None and objective.get("projection_precision", "fp32") != "fp32":
        raise ValueError("Mixed-precision recovery requires a checkpoint objective identity")


def export_prediction(model, destination, name, *, execution=None):
    """Select frozen inference output without copying the existing tensor payloads."""
    result = (execution if execution is not None else model).clone()
    if execution is not None:
        for value in list(result.graph.inputs):
            frozen = model.graph.initializers.get(value.name)
            if frozen is not None:
                result.graph.inputs.remove(value)
                value.const_value = frozen.const_value
                result.graph.register_initializer(value)
    value = ir.convenience.create_value_mapping(result.graph)[name]
    result.graph.outputs.clear()
    result.graph.outputs.append(value)
    ir.passes.common.RemoveUnusedNodesPass()(result)
    for value in list(result.graph.inputs):
        if not value.uses():
            result.graph.inputs.remove(value)
    ir.save(result, destination)
    onnx.checker.check_model(destination, full_check=True)


class DiagnosticSessions:
    """Expose the prediction for independent validation while autograd receives loss."""

    def __init__(self, factory, objective):
        self.factory, self.objective = factory, objective
        self.prediction = None

    def __call__(self, path):
        model = ir.load(path)
        prediction = self.objective["prediction"]
        values = ir.convenience.create_value_mapping(model.graph)
        model.graph.outputs.append(values[prediction])
        descriptor, destination = tempfile.mkstemp(
            prefix=".objective-", suffix=".onnx", dir=os.path.dirname(path)
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(ir.to_proto(model).SerializeToString())
            onnx.checker.check_model(destination, full_check=True)
            runtime = self.factory(destination)
        finally:
            os.unlink(destination)
        owner = self

        class Session:
            def __init__(self, inner):
                self.runtime = inner

            def run(self, names, feeds):
                result = list(self.runtime.run([*names, prediction], feeds))
                owner.prediction = result.pop()
                return result

            def close(self):
                if self.runtime is not None:
                    close = getattr(self.runtime, "close", None)
                    if close is not None:
                        close()
                    self.runtime = None

        return Session(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recompute-attention", action="store_true")
    parser.add_argument("--mlx-library")
    parser.add_argument(
        "--projection-precision", choices=["fp32", "bf16", "fp16"], default="fp32"
    )
    args = parser.parse_args()
    factory = None
    if args.mlx_library:
        from examples.lora.mlx_runtime import register_library, session

        register_library(args.mlx_library)
        factory = session
    print(
        json.dumps(
            prepare(
                args.training_run,
                args.output_dir,
                recompute_attention=args.recompute_attention,
                projection_precision=args.projection_precision,
                session_factory=factory,
            )
        )
    )


if __name__ == "__main__":
    main()
