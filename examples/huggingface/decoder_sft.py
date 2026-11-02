# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Fine-tune Mobius decoder families through the public checkpoint execution layer."""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version

try:
    import resource
except ImportError:
    resource = None

import onnx_ir as ir
import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from examples.huggingface.decoder_families import FAMILIES, prepare_graph
from examples.huggingface.decoder_graph import DECODER_BOUNDARIES
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.peft_adapter import export_adapter, roundtrip
from examples.huggingface.pretrained import load_tokenizer
from examples.huggingface.sft import make_trainer, token_loss, tokenizer
from examples.huggingface.validation import add_reference_parameters
from onnax.torch import CheckpointedModule


class OnnxDecoder(PreTrainedModel):
    """Task adapter: token interface and external loss around a reusable ONNX module."""

    config_class = PretrainedConfig
    base_model_prefix = "onnx"
    accepts_loss_kwargs = True

    def __init__(self, config, execution, mapping, family, *, head=None):
        super().__init__(config)
        # Trainer must count shifted prediction targets across accumulation groups.
        self.loss_type = "ForCausalLM"
        self.onnx = execution
        self.mapping = mapping
        self.family = family
        self.head = head
        self.mode = "chunked" if head is not None else "full"
        if head is not None:
            from examples.huggingface.decoder_head import identity

            digest = identity(head)
            self.register_buffer(
                "_head_identity", torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)
            )
            self.register_load_state_dict_pre_hook(_check_head_identity)
            self.config.onnax_head = {"identity": digest, "chunk_size": head.chunk_size}
        self.config.use_cache = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        num_items_in_batch=None,
        use_cache=False,
        **kwargs,
    ):
        if use_cache:
            raise ValueError("Training uses a cache-free full-sequence graph")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = self.onnx(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        if self.head is not None:
            if labels is not None:
                return CausalLMOutput(loss=self.head(logits, labels, num_items_in_batch))
            logits = self.head.forward(logits)
        loss = None if labels is None else token_loss(logits, labels, num_items_in_batch)
        return CausalLMOutput(loss=loss, logits=logits)

    def export(self, directory):
        if os.path.exists(directory) and os.listdir(directory):
            raise FileExistsError("Export into a new or empty directory")
        os.makedirs(directory, exist_ok=True)
        execution = os.path.join(directory, "execution")
        self.onnx.export(execution)
        record = {
            "version": 1,
            "family": self.family,
            "mapping": self.mapping,
            "base_model": self.config._name_or_path,
        }
        if self.head is not None:
            from examples.huggingface.decoder_head import export

            record["head"], record["inference"] = export(self.head, execution)
            record["version"] = 2
        self.config.save_pretrained(directory)
        with open(os.path.join(directory, "task.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

    @classmethod
    def load(cls, directory, *, session_factory=None):
        with open(os.path.join(directory, "task.json"), encoding="utf-8") as stream:
            record = json.load(stream)
        if record.get("version") not in (1, 2) or record.get("family") not in FAMILIES:
            raise ValueError("Unsupported decoder task interface")
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        if config.model_type != FAMILIES[record["family"]].config_type.model_type:
            raise ValueError("Decoder family differs from the saved configuration")
        config._name_or_path = record["base_model"]
        if source := getattr(config, "onnax_pretrained", None):
            config._commit_hash = source["revision"]
        execution = CheckpointedModule.load(
            os.path.join(directory, "execution"), session_factory=session_factory
        )
        with contextlib.ExitStack() as resources:
            resources.callback(execution.close)
            _validate_mapping(record["mapping"], execution, config)
            head = None
            if record["version"] == 2:
                from examples.huggingface.decoder_head import load, validate_file
                from examples.lora.train import session

                inference = record["inference"]
                validate_file(
                    os.path.join(directory, "execution"),
                    inference["model"],
                    inference["sha256"],
                )
                head = load(
                    os.path.join(directory, "execution"),
                    record["head"],
                    session_factory=session_factory or session,
                )
                resources.callback(head.close)
            result = cls(config, execution, record["mapping"], record["family"], head=head)
            resources.pop_all()
            return result

    def close(self):
        self.onnx.close()
        if self.head is not None:
            self.head.close()


def _check_head_identity(module, state_dict, prefix, *_args):
    key = prefix + "_head_identity"
    value = state_dict.get(key)
    if (
        value is None
        or value.dtype != torch.uint8
        or not torch.equal(value.cpu(), module._head_identity.cpu())
    ):
        raise ValueError("Checkpoint frozen-head identity differs from this model")


def _validate_mapping(mapping, execution, config):
    parameters = execution.parameters_by_name()
    names = [row[key] for row in mapping for key in ("A", "B")]
    if len(names) != len(set(names)) or set(names) != set(parameters):
        raise ValueError("Source adapter mapping differs from the parameter interface")
    for row in mapping:
        a, b = parameters[row["A"]], parameters[row["B"]]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != config.hidden_size
            or a.shape[1] != row["rank"]
            or b.shape[0] != row["rank"]
        ):
            raise ValueError("Source adapter shape/rank mapping mismatch")
        if row["source_weight"] != row["module"] + ".weight" or not math.isfinite(
            row["alpha"]
        ):
            raise ValueError("Source adapter weight/scaling mapping mismatch")
        end = 0
        for piece in row["pieces"]:
            if piece["start"] != end or piece["stop"] <= end:
                raise ValueError("Source adapter projection ranges must be contiguous")
            end = piece["stop"]
        if end != b.shape[1]:
            raise ValueError("Source adapter projection ranges differ from B")


def prepare(
    family,
    *,
    checkpoint_layers=1,
    session_factory=None,
    fused_ops=False,
    real=False,
    cache_dir=None,
    head_chunk_size=None,
):
    if (
        isinstance(checkpoint_layers, bool)
        or not isinstance(checkpoint_layers, int)
        or checkpoint_layers < 1
    ):
        raise ValueError("checkpoint_layers must be a positive integer")
    if head_chunk_size is not None and (
        isinstance(head_chunk_size, bool)
        or not isinstance(head_chunk_size, int)
        or head_chunk_size < 1
    ):
        raise ValueError("head_chunk_size must be a positive integer")
    graph, parameters, mapping, reference = prepare_graph(
        family, preserve_fused=fused_ops, real=real, cache_dir=cache_dir
    )
    if head_chunk_size is not None:
        from onnxscript import optimizer

        values = ir.convenience.create_value_mapping(graph.graph)
        graph.graph.outputs.clear()
        graph.graph.outputs.append(values["hidden_states"])
        optimizer.remove_unused_nodes(graph)
    hidden = json.loads(graph.metadata_props[DECODER_BOUNDARIES])["hidden_states"]
    boundaries = {"prefix": hidden[0]}
    for start in range(0, len(hidden) - 1, checkpoint_layers):
        stop = min(start + checkpoint_layers, len(hidden) - 1)
        boundaries[f"decoder_{start}_{stop}"] = hidden[stop]
    module = CheckpointedModule(
        graph,
        parameters,
        boundaries=boundaries,
        outputs=["hidden_states" if head_chunk_size is not None else "logits"],
        input_gradients=(),
        session_factory=session_factory,
        optimize=optimize_training_graph,
    )
    with contextlib.ExitStack() as resources:
        resources.callback(module.close)
        head = None
        if head_chunk_size is not None:
            from examples.huggingface.chunked_loss import ChunkedHead
            from examples.lora.train import session

            projection = reference.get_output_embeddings()
            if getattr(projection, "bias", None) is not None:
                raise ValueError("Decoder head expects a bias-free output projection")
            head = ChunkedHead(
                projection.weight.detach().cpu().numpy(),
                chunk_size=head_chunk_size,
                weight_layout="vocab_hidden",
                session_factory=session_factory or session,
            )
            resources.callback(head.close)
        result = OnnxDecoder(reference.config, module, mapping, family, head=head)
        resources.pop_all()
        return result, reference


def batch():
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        "labels": ids.masked_fill(mask == 0, -100),
    }


def task_tokenizer(family, *, real=False, cache_dir=None):
    if not real:
        return tokenizer()
    source = FAMILIES[family].pretrained
    if source is None:
        raise ValueError(f"No pinned pretrained source for {family}")
    return load_tokenizer(source, cache_dir=cache_dir)


def text_batch(tok):
    """Use valid tokenizer IDs, right padding and shifted loss targets for audits."""
    values = tok(
        [
            "ONNX models can learn from examples.",
            "Low-rank adapters update a small set of parameters.",
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return {
        "input_ids": values["input_ids"],
        "attention_mask": values["attention_mask"],
        "position_ids": (values["attention_mask"].cumsum(-1) - 1).clamp(min=0),
        "labels": values["input_ids"].masked_fill(values["attention_mask"] == 0, -100),
    }


def _frozen(parameters):
    return {
        name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
        for name, value in parameters
        if not value.requires_grad
    }


def audit(
    family,
    directory,
    *,
    steps=6,
    session_factory=None,
    fused_ops=False,
    adam_epsilon=1e-5,
    real=False,
    cache_dir=None,
    head_chunk_size=None,
):
    """Compare native source modules, including nonzero-B derivatives and PEFT exchange."""
    model, reference = prepare(
        family,
        session_factory=session_factory,
        fused_ops=fused_ops,
        real=real,
        cache_dir=cache_dir,
        head_chunk_size=head_chunk_size,
    )
    audit_data = (
        text_batch(task_tokenizer(family, real=True, cache_dir=cache_dir)) if real else batch()
    )
    reference = add_reference_parameters(reference, model.onnx.arrays(), model.mapping)
    frozen = _frozen(reference.named_parameters())
    optimizers = [
        torch.optim.AdamW(
            [p for p in item.parameters() if p.requires_grad],
            lr=0.003,
            weight_decay=0,
            eps=adam_epsilon,
        )
        for item in (model, reference)
    ]
    losses = []
    maximum_gradient_error = 0.0
    try:
        for step in range(steps):
            for optimizer in optimizers:
                optimizer.zero_grad()
            data = dict(audit_data)
            if step % 2:
                data = {name: value[:1, :4] for name, value in data.items()}
            actual = model(**data)
            logits = reference(
                **{name: value for name, value in data.items() if name != "labels"},
                use_cache=False,
            ).logits
            loss = token_loss(logits, data["labels"])
            predictions = actual.logits
            if predictions is None:
                with torch.no_grad():
                    predictions = model(
                        **{k: v for k, v in data.items() if k != "labels"}
                    ).logits
            torch.testing.assert_close(predictions, logits, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(actual.loss, loss, rtol=1e-4, atol=1e-6)
            actual.loss.backward()
            loss.backward()
            parameters = model.onnx.parameters_by_name()
            for row in model.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    value, expected = parameters[row[key]], getattr(target, key)
                    torch.testing.assert_close(value.grad, expected.grad, rtol=1e-4, atol=1e-6)
                    maximum_gradient_error = max(
                        maximum_gradient_error, float((value.grad - expected.grad).abs().max())
                    )
                    if step == 0 and key == "A":
                        assert value.grad.count_nonzero() == 0
                    if step == 0 and key == "B":
                        assert value.grad.count_nonzero() > 0
                    if step == 1 and key == "A":
                        assert value.grad.count_nonzero() > 0
            for optimizer in optimizers:
                optimizer.step()
            for row in model.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    torch.testing.assert_close(
                        parameters[row[key]], getattr(target, key), rtol=1e-4, atol=1e-6
                    )
            losses.append(float(actual.loss.detach()))
        assert frozen == _frozen(reference.named_parameters())
        model.export(os.path.join(directory, "export"))
        inference_data = {k: v for k, v in audit_data.items() if k != "labels"}
        expected = model(**inference_data).logits.detach()
        with_checkpoint = OnnxDecoder.load(
            os.path.join(directory, "export"), session_factory=session_factory
        )
        try:
            torch.testing.assert_close(
                with_checkpoint(**inference_data).logits, expected, rtol=0, atol=0
            )
            # A fresh native source model receives the standard PEFT adapter.
            from examples.huggingface.decoder_families import reference_model

            exchange = roundtrip(
                with_checkpoint,
                reference_model(family, real=real, cache_dir=cache_dir),
                audit_data,
                os.path.join(directory, "adapter"),
                adam_epsilon=adam_epsilon,
            )
        finally:
            with_checkpoint.close()
        return {
            "family": family,
            "independent_steps": steps,
            "adam_epsilon": adam_epsilon,
            "losses": losses,
            "max_gradient_abs_error": maximum_gradient_error,
            "frozen_source_weights_unchanged": True,
            "reload": True,
            "peft": exchange,
            "adapter_tensors": len(model.onnx.parameter_names),
        }
    finally:
        model.close()


def run(
    directory,
    *,
    family="phi3",
    steps=30,
    resume=None,
    stop=None,
    session_factory=None,
    fused_ops=False,
    adam_epsilon=1e-5,
    real=False,
    cache_dir=None,
    head_chunk_size=None,
):
    """Let TRL own training, accumulation, optimizer and checkpoint policy."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if stop is not None and (
        isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= steps
    ):
        raise ValueError("stop must select a training step")
    if os.path.exists(os.path.join(directory, "export")):
        raise FileExistsError("Select an output directory without an existing export")
    if resume is not None:
        with open(os.path.join(resume, "config.json"), encoding="utf-8") as stream:
            previous_config = json.load(stream)
        if previous_config["model_type"] != FAMILIES[family].config_type.model_type:
            raise ValueError("Checkpoint decoder family mismatch")
        if (previous_config.get("onnax_head") is not None) != (head_chunk_size is not None):
            raise ValueError("Checkpoint head execution mode mismatch")
        previous_source = previous_config.get("onnax_pretrained")
        source = FAMILIES[family].pretrained
        if real and (
            source is None
            or previous_source is None
            or previous_source["model_id"] != source.model_id
            or previous_source["revision"] != source.revision
        ):
            raise ValueError("Checkpoint pretrained source mismatch")
        if not real and previous_source is not None:
            raise ValueError("A pretrained checkpoint requires real-model execution")
    os.makedirs(directory, exist_ok=True)
    tok = task_tokenizer(family, real=real, cache_dir=cache_dir)
    model, reference = prepare(
        family,
        session_factory=session_factory,
        fused_ops=fused_ops,
        real=real,
        cache_dir=cache_dir,
        head_chunk_size=head_chunk_size,
    )
    del reference
    batch_records = []

    def record_batch(module, args, kwargs):
        ids, labels = kwargs["input_ids"], kwargs.get("labels")
        batch_records.append(
            {
                "training": module.training,
                "batch": ids.shape[0],
                "sequence": ids.shape[1],
                "valid_shifted_targets": int(labels[:, 1:].ne(-100).sum())
                if labels is not None
                else None,
            }
        )

    handle = model.register_forward_pre_hook(record_batch, with_kwargs=True)
    try:
        if len(tok) > model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds the frozen ONNX embedding")
        trainer = make_trainer(
            model,
            tok,
            directory,
            real=real,
            steps=steps,
            adam_epsilon=adam_epsilon,
            pad_to_multiple_of=64 if real else None,
        )
        trainer.args.save_steps = max(1, steps // 2)
        if stop is not None:
            from examples.huggingface.trainer import StopAt

            trainer.add_callback(StopAt(stop))
        start = time.perf_counter()
        initial_train = trainer.evaluate(trainer.train_dataset)["eval_loss"]
        initial = trainer.evaluate()["eval_loss"]
        trainer.train(resume_from_checkpoint=resume)
        final = trainer.evaluate()["eval_loss"]
        final_train = trainer.evaluate(trainer.train_dataset)["eval_loss"]
        if trainer.state.global_step == steps and (
            not final_train < initial_train or not final < initial
        ):
            raise AssertionError("Fixed training and validation losses must decrease")
        output = os.path.join(directory, "export")
        model.export(output)
        export_adapter(model, os.path.join(directory, "adapter"))
        return {
            "family": family,
            "pretrained_source": getattr(model.config, "onnax_pretrained", None),
            "steps": trainer.state.global_step,
            "microbatch_size": trainer.args.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
            "max_length": 64,
            "pad_to_multiple_of": 64 if real else None,
            "batches": batch_records,
            "adam_epsilon": adam_epsilon,
            "validation_loss": [initial, final],
            "training_loss": [initial_train, final_train],
            "source_identity": model.onnx._identity,
            "timing_scope": "initial/final evaluation, training and export; excludes construction",
            "elapsed_seconds": time.perf_counter() - start,
            "source_mapping": copy.deepcopy(model.mapping),
            "export": output,
        }
    finally:
        handle.remove()
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILIES), default="phi3")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-steps", type=int, default=6)
    parser.add_argument("--fused-ops", action="store_true")
    parser.add_argument("--head-chunk-size", type=int)
    parser.add_argument("--provider", choices=("cpu", "mlx"), default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--resume")
    parser.add_argument("--adam-epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.audit_steps < 2
        or not math.isfinite(args.adam_epsilon)
        or args.adam_epsilon <= 0
    ):
        parser.error("Require positive steps/epsilon and at least two audit steps")
    torch.set_num_threads(1)
    factory = None
    library_sha256 = None
    if args.provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if args.library:
            with open(args.library, "rb") as stream:
                library_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            register_library(args.library)
        factory = session
    elif args.library:
        parser.error("--library requires --provider mlx")
    options = {
        "session_factory": factory,
        "fused_ops": args.fused_ops,
        "adam_epsilon": args.adam_epsilon,
        "real": args.real,
        "cache_dir": args.cache_dir,
        "head_chunk_size": args.head_chunk_size,
    }
    if args.audit_only:
        report = audit(args.family, args.output_dir, steps=args.audit_steps, **options)
    else:
        report = run(
            args.output_dir,
            family=args.family,
            steps=args.steps,
            resume=args.resume,
            **options,
        )
    source_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    source_commit, working_tree_dirty = None, None
    with contextlib.suppress(OSError):
        commit = subprocess.run(
            ["git", "-C", source_directory, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", source_directory, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        source_commit = commit.stdout.strip() if commit.returncode == 0 else None
        working_tree_dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    peak_rss = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)
        if resource is not None
        else None
    )
    report.update(
        provider=args.provider,
        dtype="float32",
        library_sha256=library_sha256,
        source_commit=source_commit,
        working_tree_dirty=working_tree_dirty,
        platform=platform.platform(),
        versions={
            name: version(name)
            for name in (
                "onnax",
                "onnx",
                "onnx-ir",
                "onnxscript",
                "onnxruntime",
                "torch",
                "transformers",
                "trl",
                "datasets",
                "peft",
                "mobius-onnx",
            )
        },
        threads={"torch": torch.get_num_threads(), "ort_intra": 1, "ort_inter": 1},
        process_peak_rss_bytes=peak_rss,
        rss_scope="process, imports, construction, runtime, training or audit, and export",
    )
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
