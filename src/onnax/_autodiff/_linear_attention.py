# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Directional recurrence and chunk-recomputed gated-delta pullback."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _FLOATS, _input, _rank
from onnax._autodiff._context import _Context
from onnax._rules import _BUILTINS


class _LinearAttention:
    def __init__(self, ctx: _Context, node: ir.Node) -> None:
        if len(node.inputs) != 6 or len(node.outputs) != 2:
            raise ValueError("LinearAttention requires six inputs and two outputs")
        self.ctx = ctx
        self.values = [_input(node, i) for i in range(6)]
        q, k, v, state, decay, beta = self.values
        if q.dtype not in _FLOATS or any(x.dtype != q.dtype for x in self.values):
            raise TypeError("LinearAttention products require matching FLOAT or DOUBLE inputs")
        if any(_rank(x) != 3 for x in (q, k, v, decay, beta)) or _rank(state) != 4:
            raise ValueError(
                "LinearAttention requires rank-three sequences and rank-four state"
            )
        if set(node.attributes) - {"q_num_heads", "kv_num_heads", "scale", "update_rule"}:
            raise NotImplementedError("Unsupported LinearAttention attributes")
        if (
            "update_rule" in node.attributes
            and node.attributes["update_rule"].as_string() != "gated_delta"
        ):
            raise NotImplementedError("LinearAttention products currently require gated_delta")
        try:
            self.qh = node.attributes["q_num_heads"].as_int()
            self.vh = node.attributes["kv_num_heads"].as_int()
        except KeyError as error:
            raise ValueError("LinearAttention requires explicit head counts") from error
        if self.qh < 1 or self.vh < 1 or self.vh % self.qh:
            raise ValueError("Value heads must be a positive multiple of query/key heads")
        assert state.shape is not None
        dk, dv = state.shape[2:]
        if not isinstance(dk, int) or not isinstance(dv, int) or dk < 1 or dv < 1:
            raise ValueError("LinearAttention requires static positive state head widths")
        self.dk, self.dv = dk, dv
        assert decay.shape is not None
        if decay.shape[2] not in (self.vh, self.vh * self.dk):
            raise ValueError("LinearAttention decay must be per-head or per-key")
        self.gd = 1 if decay.shape[2] == self.vh else self.dk
        for value, width in (
            (q, self.qh * self.dk),
            (k, self.qh * self.dk),
            (v, self.vh * self.dv),
            (decay, self.vh * self.gd),
            (beta, self.vh),
        ):
            assert value.shape is not None
            if value.shape[2] != width:
                raise ValueError(
                    "LinearAttention sequence widths disagree with head dimensions"
                )
        if state.shape[1] != self.vh:
            raise ValueError("LinearAttention state head count disagrees with kv_num_heads")
        self.scale = (
            node.attributes["scale"].as_float()
            if "scale" in node.attributes
            else 1 / math.sqrt(self.dk)
        )
        if not math.isfinite(self.scale):
            raise ValueError("LinearAttention scale must be finite")

    def sequence(self, value: ir.Value, index: int) -> ir.Value:
        ctx = self.ctx
        if index == 5:
            return ctx.op("Transpose", value, perm=[1, 0, 2])
        heads = self.qh if index < 2 else self.vh
        width = self.dv if index == 2 else self.gd if index == 4 else self.dk
        shape = ctx.op(
            "Concat",
            ctx.op("Shape", value, start=0, end=2),
            ctx.constant([heads, width]),
            axis=0,
        )
        expanded = ctx.reshape(value, shape)
        if index < 2 and self.vh != self.qh:
            expanded = ctx.op("Unsqueeze", expanded, ctx.constant([3]))
            expanded = ctx.op("Tile", expanded, ctx.constant([1, 1, 1, self.vh // self.qh, 1]))
            shape = ctx.op(
                "Concat",
                ctx.op("Shape", value, start=0, end=2),
                ctx.constant([self.vh, width]),
                axis=0,
            )
            expanded = ctx.reshape(expanded, shape)
        return ctx.op("Transpose", expanded, perm=[1, 0, 2, 3])


def _outer(ctx: _Context, a: ir.Value, b: ir.Value) -> ir.Value:
    return ctx.op(
        "Mul",
        ctx.op("Unsqueeze", a, ctx.constant([-1])),
        ctx.op("Unsqueeze", b, ctx.constant([-2])),
    )


def _retrieve(ctx: _Context, state: ir.Value, key: ir.Value) -> ir.Value:
    return ctx.op(
        "Squeeze",
        ctx.op("MatMul", ctx.op("Unsqueeze", key, ctx.constant([-2])), state),
        ctx.constant([-2]),
    )


def _advance(
    ctx: _Context,
    state: ir.Value,
    key: ir.Value,
    value: ir.Value,
    decay: ir.Value,
    beta: ir.Value,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    factor = ctx.op("Exp", ctx.op("Unsqueeze", decay, ctx.constant([-1])))
    faded = ctx.op("Mul", state, factor)
    residual = ctx.op("Sub", value, _retrieve(ctx, faded, key))
    rate = ctx.op("Unsqueeze", beta, ctx.constant([-1, -2]))
    present = ctx.op("Add", faded, ctx.op("Mul", rate, _outer(ctx, key, residual)))
    return present, faded, residual, factor


def _body(
    ctx: _Context,
    a: _LinearAttention,
    stem: str,
    shapes: Sequence[Sequence[int | None]],
) -> tuple[ir.Graph, _Context]:
    prefix = ctx.name(stem)
    inputs = [
        ir.val(f"{prefix}_input_{i}", dtype=a.values[0].dtype, shape=shape)
        for i, shape in enumerate(shapes)
    ]
    graph = ir.Graph(inputs, [], nodes=[], opset_imports={"": 18}, name=prefix)
    return graph, _Context(graph)


def _finish(graph: ir.Graph, outputs: Sequence[ir.Value]) -> None:
    graph.outputs.extend(outputs)
    for index, node in enumerate(graph):
        for slot, output in enumerate(node.outputs):
            output.name = f"{graph.name}_value_{index}_{slot}"


def _checkpoint_chunk(ctx: _Context, time: ir.Value) -> ir.Value:
    # Boundary states cost ceil(T / C), replay states cost C. The minimum over
    # 32-token tiles lies on either side of sqrt(T); prefer the smaller on ties.
    root = ctx.op(
        "Cast",
        ctx.op("Ceil", ctx.op("Sqrt", ctx.op("Cast", time, to=ir.DataType.FLOAT))),
        to=ir.DataType.INT64,
    )
    tile = ctx.constant([32])
    upper = ctx.op(
        "Max",
        tile,
        ctx.op("Mul", ctx.op("Div", ctx.op("Add", root, ctx.constant([31])), tile), tile),
    )
    lower = ctx.op("Max", tile, ctx.op("Sub", upper, tile))

    def cost(chunk: ir.Value) -> ir.Value:
        numerator = ctx.op("Sub", ctx.op("Add", time, chunk), ctx.constant([1]))
        return ctx.op("Add", ctx.op("Div", numerator, chunk), chunk)

    return ctx.op("Where", ctx.op("Less", cost(upper), cost(lower)), upper, lower)


@_BUILTINS.vjp("com.microsoft", "LinearAttention", since_version=1, until_version=1)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    a = _LinearAttention(ctx, node)
    state_shape: list[int | None] = [None, a.vh, a.dk, a.dv]
    key_shape: list[int | None] = [None, a.vh, a.dk]
    value_shape: list[int | None] = [None, a.vh, a.dv]
    decay_shape: list[int | None] = [None, a.vh, a.gd]
    beta_shape: list[int | None] = [None, a.vh]
    # Forward chunks retain only their starting state. Replay reconstructs a
    # single chunk's previous states for its reverse recurrence.
    forward_step, forward = _body(
        ctx,
        a,
        "linear_attention_forward_step",
        [state_shape, key_shape, value_shape, decay_shape, beta_shape],
    )
    state, key, value, decay, beta = forward_step.inputs
    present, _, _, _ = _advance(forward, state, key, value, decay, beta)
    _finish(forward_step, [present])
    replay_step = forward_step.clone(deep_copy=True)
    replay_step.name = ctx.name("linear_attention_replay_step")
    replay_step.outputs.append(
        ir.tape.Tape(replay_step).op("Identity", [replay_step.inputs[0]])
    )
    for index, argument in enumerate(replay_step.inputs):
        argument.name = f"{replay_step.name}_input_{index}"
    for index, operation in enumerate(replay_step):
        for slot, output in enumerate(operation.outputs):
            output.name = f"{replay_step.name}_value_{index}_{slot}"

    chunk_shapes = [
        [None, *shape] for shape in (key_shape, value_shape, decay_shape, beta_shape)
    ]
    forward_chunk, forward = _body(
        ctx, a, "linear_attention_forward_chunk", [state_shape, *chunk_shapes]
    )
    (last,) = forward.tape.op_multi_out(
        "Scan",
        list(forward_chunk.inputs),
        {"body": forward_step, "num_scan_inputs": 4},
        num_outputs=1,
    )
    _finish(forward_chunk, [last, forward.op("Identity", forward_chunk.inputs[0])])

    reverse_step, reverse = _body(
        ctx,
        a,
        "linear_attention_reverse_step",
        [
            state_shape,
            state_shape,
            key_shape,
            key_shape,
            value_shape,
            decay_shape,
            beta_shape,
            value_shape,
        ],
    )
    carry, previous, q, k, v, g, beta, gy = reverse_step.inputs
    present, faded, residual, factor = _advance(reverse, previous, k, v, g, beta)
    scaled_q = reverse.op("Mul", q, reverse.scalar(q, a.scale))
    total = reverse.op("Add", carry, _outer(reverse, scaled_q, gy))

    def matrix_vector(matrix, vector):
        return reverse.op(
            "Squeeze",
            reverse.op(
                "MatMul", matrix, reverse.op("Unsqueeze", vector, reverse.constant([-1]))
            ),
            reverse.constant([-1]),
        )

    dq = reverse.op("Mul", matrix_vector(present, gy), reverse.scalar(q, a.scale))
    product = _retrieve(reverse, total, k)
    dr = reverse.op("Mul", product, reverse.op("Unsqueeze", beta, reverse.constant([-1])))
    db = reverse.op(
        "ReduceSum", reverse.op("Mul", product, residual), reverse.constant([-1]), keepdims=0
    )
    dk = reverse.op(
        "Sub",
        reverse.op(
            "Mul",
            matrix_vector(total, residual),
            reverse.op("Unsqueeze", beta, reverse.constant([-1])),
        ),
        matrix_vector(faded, dr),
    )
    dfaded = reverse.op("Sub", total, _outer(reverse, k, dr))
    dg = reverse.op(
        "ReduceSum", reverse.op("Mul", dfaded, faded), reverse.constant([-1]), keepdims=0
    )
    dprevious = reverse.op("Mul", factor, dfaded)
    if a.gd == 1:
        dg = reverse.op("ReduceSum", dg, reverse.constant([-1]), keepdims=1)
    _finish(reverse_step, [dprevious, dq, dk, dr, dg, db])

    reverse_chunk, reverse = _body(
        ctx,
        a,
        "linear_attention_reverse_chunk",
        [
            state_shape,
            state_shape,
            *[
                [None, *shape]
                for shape in (
                    key_shape,
                    key_shape,
                    value_shape,
                    decay_shape,
                    beta_shape,
                    value_shape,
                )
            ],
        ],
    )
    carry, boundary, qc, kc, vc, gc, bc, gyc = reverse_chunk.inputs
    _, history = reverse.tape.op_multi_out(
        "Scan",
        [boundary, kc, vc, gc, bc],
        {"body": replay_step, "num_scan_inputs": 4},
        num_outputs=2,
    )
    outputs = reverse.tape.op_multi_out(
        "Scan",
        [carry, history, qc, kc, vc, gc, bc, gyc],
        {
            "body": reverse_step,
            "num_scan_inputs": 7,
            "scan_input_directions": [1] * 7,
            "scan_output_directions": [1] * 5,
        },
        num_outputs=6,
    )
    _finish(reverse_chunk, outputs)

    time = ctx.op("Shape", a.values[0], start=1, end=2)
    chunk_size = _checkpoint_chunk(ctx, time)
    padding = ctx.op(
        "Mod",
        ctx.op("Sub", chunk_size, ctx.op("Mod", time, chunk_size)),
        chunk_size,
    )

    def chunks(sequence, rank):
        pads = ctx.op(
            "Concat", ctx.constant([0] * rank), padding, ctx.constant([0] * (rank - 1)), axis=0
        )
        padded = ctx.op("Pad", sequence, pads)
        shape = ctx.op(
            "Concat",
            ctx.constant([-1]),
            chunk_size,
            ctx.op("Shape", sequence, start=1),
            axis=0,
        )
        return ctx.reshape(padded, shape)

    sequences = [a.sequence(a.values[index], index) for index in (0, 1, 2, 4, 5)]
    qc, kc, vc, gc = [chunks(value, 4) for value in sequences[:4]]
    bc = chunks(sequences[4], 3)
    gyc = chunks(a.sequence(gs[0], 2), 4)
    _, boundaries = ctx.tape.op_multi_out(
        "Scan",
        [a.values[3], kc, vc, gc, bc],
        {"body": forward_chunk, "num_scan_inputs": 4},
        num_outputs=2,
    )
    dstate, *gradients = ctx.tape.op_multi_out(
        "Scan",
        [gs[1], boundaries, qc, kc, vc, gc, bc, gyc],
        {
            "body": reverse_chunk,
            "num_scan_inputs": 7,
            "scan_input_directions": [1] * 7,
            "scan_output_directions": [1] * 5,
        },
        num_outputs=6,
    )
    results = {3: dstate}
    for index, gradient in zip((0, 1, 2, 4, 5), gradients, strict=True):
        width = a.dv if index == 2 else a.gd if index == 4 else a.dk
        tail = [a.vh] if index == 5 else [a.vh, width]
        batch = ctx.op("Shape", a.values[0], start=0, end=1)
        shape = ctx.op("Concat", ctx.constant([-1]), batch, ctx.constant(tail), axis=0)
        gradient = ctx.reshape(gradient, shape)
        gradient = ctx.op("Slice", gradient, ctx.constant([0]), time, ctx.constant([0]))
        if index != 5:
            gradient = ctx.op("Transpose", gradient, perm=[1, 0, 2, 3])
            if index < 2 and a.qh != a.vh:
                shape = ctx.op(
                    "Concat", batch, time, ctx.constant([a.qh, a.vh // a.qh, a.dk]), axis=0
                )
                gradient = ctx.op(
                    "ReduceSum", ctx.reshape(gradient, shape), ctx.constant([3]), keepdims=0
                )
        else:
            gradient = ctx.op("Transpose", gradient, perm=[1, 0, 2])
        results[index] = ctx.reshape(gradient, ctx.op("Shape", a.values[index]))
    return {index: results[index] for index in indices}


@_BUILTINS.jvp("com.microsoft", "LinearAttention", since_version=1, until_version=1)
def jvp(ctx: _Context, node: ir.Node, tangents: Mapping[int, ir.Value]) -> Sequence[ir.Value]:
    a = _LinearAttention(ctx, node)
    prefix = ctx.name("linear_attention_jvp")
    dtype = a.values[0].dtype
    assert dtype is not None
    shapes: list[list[int | None]] = [
        [None, a.vh, a.dk, a.dv],
        [None, a.vh, a.dk, a.dv],
        *([[None, a.vh, a.dk]] * 2),
        [None, a.vh, a.dv],
        [None, a.vh, a.gd],
        [None, a.vh],
        *([[None, a.vh, a.dk]] * 2),
        [None, a.vh, a.dv],
        [None, a.vh, a.gd],
        [None, a.vh],
    ]
    inputs = [
        ir.val(f"{prefix}_input_{i}", dtype=dtype, shape=shape)
        for i, shape in enumerate(shapes)
    ]
    body = ir.Graph(inputs, [], nodes=[], opset_imports={"": 18}, name=prefix)
    step = _Context(body)
    state, ds, q, k, v, g, beta, dq, dk, dv, dg, db = inputs
    factor = step.op("Exp", step.op("Unsqueeze", g, step.constant([-1])))
    faded = step.op("Mul", factor, state)
    dfaded = step.op(
        "Mul",
        factor,
        step.op(
            "Add", ds, step.op("Mul", state, step.op("Unsqueeze", dg, step.constant([-1])))
        ),
    )
    residual = step.op("Sub", v, _retrieve(step, faded, k))
    dr = step.op(
        "Sub",
        dv,
        step.op("Add", _retrieve(step, dfaded, k), _retrieve(step, faded, dk)),
    )
    rate = step.op("Unsqueeze", beta, step.constant([-1, -2]))
    drate = step.op("Unsqueeze", db, step.constant([-1, -2]))
    present = step.op("Add", faded, step.op("Mul", rate, _outer(step, k, residual)))
    differential = step.sum(
        [
            dfaded,
            step.op("Mul", drate, _outer(step, k, residual)),
            step.op("Mul", rate, _outer(step, dk, residual)),
            step.op("Mul", rate, _outer(step, k, dr)),
        ]
    )
    dy = step.op(
        "Mul",
        step.scalar(q, a.scale),
        step.op("Add", _retrieve(step, differential, q), _retrieve(step, present, dq)),
    )
    body.outputs.extend([present, differential, dy])
    for i, operation in enumerate(body):
        for slot, output in enumerate(operation.outputs):
            output.name = f"{prefix}_value_{i}_{slot}"
    primal_sequences, tangent_sequences = [], []
    for index in (0, 1, 2, 4, 5):
        primal_sequences.append(a.sequence(a.values[index], index))
        direction = tangents[index] if index in tangents else ctx.like(a.values[index])
        tangent_sequences.append(a.sequence(direction, index))
    _, ds_final, dy_sequence = ctx.tape.op_multi_out(
        "Scan",
        [
            a.values[3],
            tangents[3] if 3 in tangents else ctx.like(a.values[3]),
            *primal_sequences,
            *tangent_sequences,
        ],
        {"body": body, "num_scan_inputs": 10},
        num_outputs=3,
    )
    dy_sequence = ctx.op("Transpose", dy_sequence, perm=[1, 0, 2, 3])
    return ctx.reshape(dy_sequence, ctx.op("Shape", a.values[2])), ds_final
