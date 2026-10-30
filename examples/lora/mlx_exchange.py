# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""LoRA interface validation over the shared native MLX binding."""

from examples.lora.binding import gradient_shapes
from examples.lora.model import PARAMETERS
from examples.mlx_binding import CpuView as CpuView
from examples.mlx_binding import MlxBinding
from examples.mlx_binding import synchronize as synchronize


class MlxDLPack(MlxBinding):
    """Validate the LoRA interface and share storage through the generic binding."""

    def __init__(self, executor, interface):
        self.shapes, self.output_shapes = gradient_shapes(executor, interface)
        inputs = {value.name: value for value in executor.get_inputs()}
        self.batch = inputs["features"].shape[0]
        if not isinstance(self.batch, int) or self.batch < 1:
            raise ValueError("This native workload requires a fixed positive batch")
        if inputs["target"].shape[0] != self.batch:
            raise ValueError("Features and target metadata need the same fixed batch")
        outputs = {value.name: value for value in executor.get_outputs()}
        self.prediction_name = interface.get("prediction", "prediction")
        if self.prediction_name is not None:
            prediction = outputs.get(self.prediction_name)
            if (
                prediction is None
                or prediction.type != "tensor(float)"
                or prediction.shape != [self.batch, self.shapes["target"][1]]
            ):
                raise ValueError("Expected FP32 prediction output with the fixed batch")
            self.output_shapes = {
                self.prediction_name: (self.batch, self.shapes["target"][1]),
                **self.output_shapes,
            }
        self.loss_name = interface["loss"]
        self.gradient_names = {
            row["input"]: row["gradient"] for row in interface["parameters"]
        }
        super().__init__(executor, outputs=self.output_shapes)

    def bind(self, features, target, params):
        self.ready = False
        self.output_values = None
        if params.keys() != set(PARAMETERS):
            raise ValueError("Expected exactly lora.A and lora.B parameters")
        super().bind({"features": features, "target": target, **params})

    def import_outputs(self):
        arrays = super().import_outputs()
        return (
            arrays.get(self.prediction_name),
            arrays[self.loss_name],
            {name: arrays[output] for name, output in self.gradient_names.items()},
        )
