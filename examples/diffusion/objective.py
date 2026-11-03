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

from examples.broadcast_adjoint import fuse_broadcast_adjoints
from examples.huggingface.attention_vjp import fuse_attention_vjp
from onnax.torch import OnnxModule
from onnax.torch._serialization import file_digest


def prepare(training_run, destination):
    """Compose MSE with a qualified source; export independently loadable sessions."""
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
        return optimizer.optimize(
            fuse_broadcast_adjoints(fuse_attention_vjp(model)), inline=False
        )

    with OnnxModule(
        source, initial, input_gradients=[], eager_gradients=True, optimize=optimize
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
    return record


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
    args = parser.parse_args()
    print(json.dumps(prepare(args.training_run, args.output_dir)))


if __name__ == "__main__":
    main()
