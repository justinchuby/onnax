# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Half projection compute through frozen UNet blocks and FP32 adapter updates."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnxscript import optimizer

pytest.importorskip("mobius")
pytest.importorskip("diffusers")
torch = pytest.importorskip("torch")
from diffusers import UNet2DConditionModel

import onnax
from examples.diffusion.adapters import inject
from examples.diffusion.train import frozen_hashes
from examples.diffusion.unet import build, tiny_config
from examples.diffusion.unet_reference import make_reference
from examples.huggingface.mixed_precision import precision_projections
from examples.lora.bf16_rules import DOMAIN, matmul_function
from examples.torch_precision import projection_product
from onnax.torch import OnnxModule


def _half_path_close(actual, expected, precision):
    """Use one compute-format epsilon on the complete tensor norm."""
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    error = torch.linalg.vector_norm((actual.detach() - expected.detach()).float())
    norm = torch.linalg.vector_norm(expected.detach().float())
    assert error <= torch.finfo(dtype).eps * norm + 1e-6


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_half_projection_gradients_updates_and_reload(tmp_path, native, precision):
    if native and os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    factory = ReferenceEvaluator
    if native:
        from examples.lora.mlx_runtime import register_library, session

        if library := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
            register_library(library)
        factory = session
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    config = tiny_config()
    original = UNet2DConditionModel(**config).eval()
    graph, params, mapping = inject(build(config, original.state_dict()))
    frozen = frozen_hashes(graph)
    rng = np.random.default_rng(43)
    for name in params:
        if name.endswith(".B"):
            params[name] = rng.normal(0, 0.02, params[name].shape).astype(np.float32)
    reference, expected_params = make_reference(
        config, original.state_dict(), params, mapping, precision=precision
    )
    lowered = precision_projections(graph, precision=precision)
    batch = dict(
        sample=torch.randn(1, 4, 8, 12),
        timestep=torch.tensor([317.0]),
        encoder_hidden_states=torch.randn(1, 3, 24),
    )
    target = torch.randn(1, 4, 8, 12)
    with OnnxModule(
        lowered,
        params,
        input_gradients=[],
        optimize=optimizer.optimize,
        session_factory=factory,
    ) as model:
        update = torch.optim.AdamW(model.parameters(), lr=0.003, eps=1e-5, weight_decay=0)
        control = torch.optim.AdamW(
            expected_params.values(), lr=0.003, eps=1e-5, weight_decay=0
        )
        losses, reference_losses = [], []
        for _ in range(3):
            update.zero_grad()
            control.zero_grad()
            actual = model(**batch)
            expected = reference(**batch).sample
            assert actual.dtype == expected.dtype == torch.float32
            _half_path_close(actual, expected, precision)
            loss = (actual - target).square().mean()
            expected_loss = (expected - target).square().mean()
            torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)
            losses.append(float(loss.detach()))
            reference_losses.append(float(expected_loss.detach()))
            loss.backward()
            expected_loss.backward()
            for name, parameter in model.parameters_by_name().items():
                assert parameter.dtype == parameter.grad.dtype == torch.float32
                assert torch.isfinite(parameter.grad).all()
                if torch.count_nonzero(expected_params[name].grad):
                    assert torch.count_nonzero(parameter.grad)
            _half_path_close(
                torch.cat([p.grad.flatten() for p in model.parameters()]),
                torch.cat(
                    [
                        expected_params[name].grad.flatten()
                        for name in model.parameters_by_name()
                    ]
                ),
                precision,
            )
            update.step()
            control.step()
            for state in update.state.values():
                assert state["exp_avg"].dtype == torch.float32
                assert state["exp_avg_sq"].dtype == torch.float32
            _half_path_close(
                torch.cat([p.flatten() for p in model.parameters()]),
                torch.cat(
                    [expected_params[name].flatten() for name in model.parameters_by_name()]
                ),
                precision,
            )
        assert losses[-1] < losses[0]
        assert reference_losses[-1] < reference_losses[0]
        with torch.no_grad():
            prediction = model(**batch)
        exported = model.export(tmp_path / "trained")
        assert frozen_hashes(graph) == frozen
        assert all(frozen_hashes(exported)[name] == digest for name, digest in frozen.items())
    with (
        OnnxModule.load(tmp_path / "trained", session_factory=factory) as restored,
        torch.no_grad(),
    ):
        torch.testing.assert_close(restored(**batch), prediction, rtol=0, atol=0)
    onnx.checker.check_model(str(tmp_path / "trained" / "model.onnx"), full_check=True)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_shared_projection_inputs_keep_fp32_adjoint_contract(native, precision, device):
    if device == "mps" and (
        os.environ.get("ONNAX_TEST_MLX_EP") != "1" or not torch.backends.mps.is_available()
    ):
        pytest.skip("MPS comparison requires native opt-in and Apple GPU")
    if native and os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")

    def factory(model):
        return ReferenceEvaluator(ir.to_proto(model))

    if native:
        from examples.lora.mlx_runtime import register_library, session

        if library := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
            register_library(library)
        factory = session
    a, b = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        for name, shape in (("a", [5, 7]), ("b", [7, 3]))
    ]
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[5, 3])
    function = matmul_function(23, precision=precision)
    node = ir.node(function.name, [a, b], outputs=[y], domain=DOMAIN)
    model = ir.Model(
        ir.Graph(
            [a, b],
            [y],
            nodes=[node],
            opset_imports={"": 23, DOMAIN: 1},
            name="shared_half_inputs",
        ),
        functions=[function],
        ir_version=10,
    )
    derivative = onnax.vjp(model, inputs=["a", "b"])
    torch.manual_seed(31)
    left = torch.randn(5, 7, requires_grad=True)
    right = torch.randn(7, 3, requires_grad=True)
    g = torch.randn(5, 3)
    expected_left = left.detach().to(device).requires_grad_()
    expected_right = right.detach().to(device).requires_grad_()
    expected = projection_product(precision).apply(expected_left, expected_right)
    (expected * g.to(device)).sum().backward()
    actual = factory(derivative).run(
        None,
        {
            "a": left.detach().numpy(),
            "b": right.detach().numpy(),
            derivative.graph.inputs[-1].name: g.numpy(),
        },
    )
    for value, reference in zip(
        actual,
        (expected.detach(), expected_left.grad, expected_right.grad),
        strict=True,
    ):
        torch.testing.assert_close(
            torch.from_numpy(value), reference.cpu(), rtol=1e-4, atol=1e-6
        )
