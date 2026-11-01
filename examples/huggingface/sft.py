# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mobius Qwen3 ONNX fine-tuning through TRL SFTTrainer."""

import argparse
import functools
import gc
import hashlib
import json
import math
import os
import tempfile

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Qwen3Config,
    TrainerCallback,
)
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTConfig, SFTTrainer

from examples.huggingface.bridge import OnnxModule, frozen_hashes, graph_session
from examples.huggingface.qwen import (
    MODEL_ID,
    REVISION,
    build_base,
    decompose,
    inject,
    reference_model,
)
from examples.lora.train import session


def texts(path=None):
    path = os.path.join(os.path.dirname(__file__), "corpus.txt") if path is None else path
    with open(path, encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def tokenizer(real=False, *, training_texts=None):
    if real:
        result = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        result.pad_token = result.eos_token
        result.padding_side = "right"
        return result
    backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(
        texts() if training_texts is None else training_texts,
        trainers.WordLevelTrainer(
            vocab_size=256,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        ),
    )
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=64,
        padding_side="right",
    )


def _tokenizer_digest(tok):
    backend = json.loads(tok.backend_tokenizer.to_str())
    # Hash encoding semantics independently of per-batch padding and truncation.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    identity = {
        "backend": backend,
        "special_tokens": tok.special_tokens_map,
        "padding_side": tok.padding_side,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def token_loss(logits, labels, num_items=None):
    targets = labels[:, 1:].reshape(-1)
    total = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        targets,
        ignore_index=-100,
        reduction="sum",
    )
    count = targets.ne(-100).sum() if num_items is None else torch.as_tensor(num_items)
    return total / count.clamp(min=1)


class OnnxQwen3(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "onnx"
    accepts_loss_kwargs = True

    def __init__(
        self,
        config,
        model,
        params,
        mapping,
        *,
        mode="full",
        chunk_size=128,
        optimize=True,
        save_activations=False,
        session_factory=session,
    ):
        super().__init__(config)
        self.loss_type = "ForCausalLM"
        if mode not in ("full", "chunked", "joint"):
            raise ValueError("Expected full, chunked or joint training")
        self.mode = mode
        self.forward_graph = model
        if mode == "joint":
            from examples.huggingface.joint_loss import JointLoss, causal_loss_graph

            self._joint_session_factory = session_factory
            self.onnx = JointLoss(
                causal_loss_graph(model),
                params,
                optimize=optimize,
                session_factory=session_factory,
            )
        elif mode == "chunked":
            from onnxscript import optimizer

            from examples.huggingface.chunked_loss import ChunkedHead

            rhs = model.graph.outputs[0].producer().inputs[1]
            if rhs.const_value is not None:
                weight = rhs.const_value.numpy()
            elif rhs.producer().op_type == "Transpose":
                weight = rhs.producer().inputs[0].const_value.numpy().T
            else:
                raise ValueError("Expected a frozen output projection")
            self.head = ChunkedHead(
                weight, chunk_size=chunk_size, session_factory=session_factory
            )
            body = model.clone()
            hidden = body.graph.outputs[1]
            body.graph.outputs.clear()
            body.graph.outputs.append(hidden)
            optimizer.remove_unused_nodes(body)
            self.onnx = OnnxModule(
                body,
                params,
                output=hidden.name,
                optimize=optimize,
                save_activations=save_activations,
                session_factory=session_factory,
            )
        else:
            self.onnx = OnnxModule(
                model,
                params,
                output="logits",
                optimize=optimize,
                save_activations=save_activations,
                session_factory=session_factory,
            )
        self.mapping = mapping
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
            raise ValueError("The training graph has no KV cache")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if self.mode == "joint":
            if labels is not None:
                total = self.onnx(**feeds, labels=labels)
                count = (
                    labels[:, 1:].ne(-100).sum()
                    if num_items_in_batch is None
                    else torch.as_tensor(num_items_in_batch)
                )
                return CausalLMOutput(loss=total / count.clamp(min=1))
            # Prediction sessions are short-lived; training reuses its sole loss/gradient session.
            with tempfile.TemporaryDirectory(prefix="onnax-joint-predict-") as directory:
                executor = graph_session(
                    self.forward_graph,
                    directory,
                    "forward",
                    session_factory=self._joint_session_factory,
                )
                arrays = {
                    name: value.detach().contiguous().numpy() for name, value in feeds.items()
                }
                logits = executor.run(["logits"], {**arrays, **self.onnx.arrays()})[0]
            return CausalLMOutput(logits=torch.from_numpy(logits))
        value = self.onnx(**feeds)
        if self.mode == "chunked":
            if labels is not None:
                return CausalLMOutput(loss=self.head(value, labels, num_items_in_batch))
            logits = self.head.forward(value)
        else:
            logits = value
        loss = None if labels is None else token_loss(logits, labels, num_items_in_batch)
        return CausalLMOutput(loss=loss, logits=logits)


def _prepare_graph(real=False, *, nf4=False, projection_precision="fp32"):
    """Build one canonical Mobius source and matching reconstructed reference."""
    if projection_precision not in ("fp32", "bf16"):
        raise ValueError("Expected fp32 or bf16 projection precision")
    reference = reference_model(real)
    base = build_base(reference)
    adapted, params, mapping = inject(base)
    graph = decompose(adapted)
    quantization = None
    if nf4:
        from examples.huggingface.nf4 import lower_decoder

        graph, quantization = lower_decoder(graph, reference, precision=projection_precision)
    if projection_precision == "bf16":
        from examples.huggingface.mixed_precision import bf16_projections

        graph = bf16_projections(graph)
    return reference, graph, params, mapping, quantization


def prepare(
    real=False,
    *,
    mode="full",
    chunk_size=128,
    optimize=True,
    save_activations=False,
    nf4=False,
    projection_precision="fp32",
    session_factory=session,
):
    if projection_precision not in ("fp32", "bf16"):
        raise ValueError("Expected fp32 or bf16 projection precision")
    if projection_precision == "bf16" and mode != "joint":
        raise ValueError("BF16 projection training uses joint execution")
    reference, graph, params, mapping, quantization = _prepare_graph(
        real, nf4=nf4, projection_precision=projection_precision
    )
    model = OnnxQwen3(
        reference.config,
        graph,
        params,
        mapping,
        mode=mode,
        chunk_size=chunk_size,
        optimize=optimize,
        save_activations=save_activations,
        session_factory=session_factory,
    )
    model.quantization = quantization
    model.projection_precision = projection_precision
    return model, reference


class OnnxSFTTrainer(SFTTrainer):
    """Use the ONNX chunked loss without recomputing logits for TRL metrics."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.mode == "full":
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        from transformers import Trainer

        return Trainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)


def _training_texts(train_file=None, validation_file=None):
    if (train_file is None) != (validation_file is None):
        raise ValueError("Provide both training and validation text files")
    if train_file is None:
        lines = texts()
        train, validation = lines[:-4] * 8, lines[-4:]
    else:
        train, validation = texts(train_file), texts(validation_file)
    if not train or not validation:
        raise ValueError("Training and validation text must both be nonempty")
    return train, validation


def _validate_max_length(max_length, *, limit=None):
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("max_length must be an integer >= 2")
    if limit is not None and max_length > limit:
        raise ValueError(f"max_length exceeds the model context limit of {limit}")


def _data_identity(train, validation, tok, *, max_length=64):
    return {
        "version": 1,
        "train_lines": len(train),
        "validation_lines": len(validation),
        "train_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
        "validation_sha256": hashlib.sha256("\n".join(validation).encode()).hexdigest(),
        "tokenizer_sha256": _tokenizer_digest(tok),
        "max_length": max_length,
    }


class _DataIdentityCallback(TrainerCallback):
    def __init__(self, identity):
        self.identity = identity

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}", "onnax_data.json"
            )
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(self.identity, stream, indent=2)
        return control


def make_trainer(
    model,
    tok,
    directory,
    *,
    real=False,
    steps=None,
    callbacks=None,
    train_texts=None,
    eval_texts=None,
    max_length=64,
):
    _validate_max_length(max_length, limit=model.config.max_position_embeddings)
    if (train_texts is None) != (eval_texts is None):
        raise ValueError("Provide both training and evaluation texts")
    if train_texts is None:
        train_texts, eval_texts = _training_texts()
    if not train_texts or not eval_texts:
        raise ValueError("Training and evaluation texts must both be nonempty")
    args = SFTConfig(
        output_dir=directory,
        use_cpu=True,
        max_steps=steps or (30 if real else 100),
        per_device_train_batch_size=1 if real else 4,
        gradient_accumulation_steps=4 if real else 2,
        per_device_eval_batch_size=1,
        learning_rate=0.003 if real else 0.01,
        optim="adamw_torch",
        weight_decay=0,
        max_grad_norm=0,
        lr_scheduler_type="constant",
        prediction_loss_only=True,
        max_length=max_length,
        packing=False,
        gradient_checkpointing=False,
        dataset_num_proc=None,
        report_to="none",
        logging_strategy="no",
        save_strategy="steps",
        save_steps=15 if real else 50,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=20261029,
    )
    return OnnxSFTTrainer(
        model=model,
        args=args,
        processing_class=tok,
        train_dataset=Dataset.from_dict({"text": train_texts}),
        eval_dataset=Dataset.from_dict({"text": eval_texts}),
        callbacks=[
            *(callbacks or ()),
            _DataIdentityCallback(
                _data_identity(train_texts, eval_texts, tok, max_length=max_length)
            ),
        ],
    )


def run(
    directory,
    *,
    real=False,
    steps=None,
    resume=None,
    mode="full",
    chunk_size=128,
    optimize=True,
    diagnose_rounding=None,
    save_activations=False,
    fp64_audit=False,
    nf4=False,
    peft_roundtrip=False,
    provider="cpu",
    library=None,
    segment_nodes=None,
    stable_reference=False,
    projection_precision="fp32",
    train_file=None,
    validation_file=None,
    max_length=64,
    reference_audit=False,
):
    reference_audit = bool(
        reference_audit
        or diagnose_rounding is not None
        or peft_roundtrip
        or fp64_audit
        or stable_reference
    )
    diagnose_rounding = True if diagnose_rounding is None else diagnose_rounding
    _validate_max_length(max_length, limit=None if real else 64)
    if provider not in ("cpu", "mlx", "reference"):
        raise ValueError("Expected cpu, mlx or reference provider")
    if provider != "mlx" and (library is not None or segment_nodes is not None):
        raise ValueError("Library and segmentation options require the mlx provider")
    runtime = session
    if provider == "reference":
        from examples.huggingface.serialization import reference_session

        runtime = reference_session
    if provider == "mlx":
        from examples.lora.mlx_runtime import register_library
        from examples.lora.mlx_runtime import session as mlx_session

        if library is not None:
            register_library(library)
        runtime = functools.partial(mlx_session, target_nodes=segment_nodes)
    if fp64_audit and nf4:
        raise ValueError(
            "FP64 oracle expects an FP32 base; use the dequantized FP32 NF4 reference"
        )
    if projection_precision == "bf16" and fp64_audit:
        raise ValueError("FP64 diagnostics require the FP32 projection policy")
    if projection_precision == "bf16" and provider == "cpu":
        raise ValueError("Select mlx or reference for BF16 projection execution")
    train_lines, validation_lines = _training_texts(train_file, validation_file)
    tok = tokenizer(real, training_texts=train_lines if train_file is not None else None)
    data_identity = _data_identity(train_lines, validation_lines, tok, max_length=max_length)
    if resume is True:
        from transformers.trainer_utils import get_last_checkpoint

        resume = get_last_checkpoint(directory) if os.path.isdir(directory) else None
        if resume is None:
            raise ValueError("No checkpoint found to resume")
    if resume:
        path = os.path.join(resume, "onnax_data.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                if json.load(stream) != data_identity:
                    raise ValueError(
                        "Checkpoint training data, tokenizer or context length differs"
                    )
        elif train_file is not None or max_length != 64:
            raise ValueError(
                "Custom-text or context resume requires checkpoint data identity metadata"
            )
    torch.set_num_threads(4 if real else 1)
    model, reference = prepare(
        real,
        mode=mode,
        chunk_size=chunk_size,
        optimize=optimize,
        save_activations=save_activations,
        nf4=nf4,
        projection_precision=projection_precision,
        session_factory=runtime,
    )
    _validate_max_length(max_length, limit=model.config.max_position_embeddings)
    initial_frozen = frozen_hashes(model.forward_graph)
    from examples.huggingface.validation import add_reference_adapters, audit

    initial_audit = sample = None
    if reference_audit:
        reference = add_reference_adapters(reference, model)
        if projection_precision == "bf16":
            from examples.huggingface.mixed_precision import reference_projections

            reference = reference_projections(reference)
        sample = tok(
            train_lines[0], truncation=True, max_length=max_length, return_tensors="pt"
        )
        sample = {k: v for k, v in sample.items() if k in ("input_ids", "attention_mask")}
        sample["labels"] = sample["input_ids"].clone()
        initial_audit = audit(model, reference, sample, diagnose_rounding=diagnose_rounding)
    else:
        reference = None
        gc.collect()
    initial_params = model.onnx.arrays() if fp64_audit else None

    def stable_control():
        from examples.huggingface.reference_head import pairwise_head
        from examples.huggingface.reference_losses import stable_token_loss

        with pairwise_head(reference, projection_precision=projection_precision):
            return audit(
                model,
                reference,
                sample,
                diagnose_rounding=True,
                reference_loss=stable_token_loss,
            )

    initial_stable = stable_control() if stable_reference else None
    trainer = make_trainer(
        model,
        tok,
        directory,
        real=real,
        steps=steps,
        train_texts=train_lines,
        eval_texts=validation_lines,
        max_length=max_length,
    )
    before = trainer.evaluate()["eval_loss"]
    if not math.isfinite(before):
        raise FloatingPointError("Nonfinite validation loss before training")
    trainer.train(resume_from_checkpoint=resume)
    if any(not torch.isfinite(parameter.detach()).all() for parameter in model.onnx.adapters):
        raise FloatingPointError("Nonfinite adapter parameter after training")
    after = trainer.evaluate()["eval_loss"]
    if not math.isfinite(after):
        raise FloatingPointError("Nonfinite validation loss after training")
    final_audit = (
        audit(model, reference, sample, diagnose_rounding=diagnose_rounding)
        if reference_audit
        else None
    )
    final_stable = stable_control() if stable_reference else None
    model.onnx.export(os.path.join(directory, "export"), forward_model=model.forward_graph)
    from examples.huggingface.peft_adapter import export_adapter

    export_adapter(model, os.path.join(directory, "adapter"))
    tok.save_pretrained(os.path.join(directory, "export"))
    if model.quantization is not None:
        with open(os.path.join(directory, "export", "quantization.json"), "w") as stream:
            json.dump(model.quantization, stream, indent=2)
    report = {
        "model": MODEL_ID if real else "tiny-qwen3",
        "revision": REVISION if real else None,
        "steps": trainer.state.global_step,
        "data": data_identity,
        "mode": mode,
        "projection_precision": projection_precision,
        "validation_loss": [before, after],
        "audits": [initial_audit, final_audit] if reference_audit else [],
        "reference_retained_during_training": reference_audit,
        "quantization": model.quantization,
        "provider": provider,
        "segment_target_nodes": segment_nodes,
        "ort_graph_optimization": None if provider == "reference" else "disabled",
        "frozen_buffers_unchanged": frozen_hashes(model.forward_graph) == initial_frozen,
    }
    if provider == "mlx":
        report["cpu_fallback_disabled"] = True
        report["mlx_compile_disabled"] = os.environ.get(
            "ONNXRUNTIME_EP_MLX_NO_COMPILE", ""
        ) not in ("", "0")
        report["lut4_fusion_disabled"] = os.environ.get(
            "ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION", ""
        ) not in ("", "0")
        report["library"] = os.path.abspath(library) if library else "installed plugin"
        if library:
            with open(library, "rb") as stream:
                report["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if stable_reference:
        report["stable_reference_audits"] = {
            "loss": "target-excluded stable CE",
            "head_backward": (
                "FP32 pairwise K blocks of 128 using BF16-rounded weights"
                if projection_precision == "bf16"
                else "FP32 pairwise K blocks of 128"
            ),
            "checks": [initial_stable, final_stable],
        }
    final_params = model.onnx.arrays()
    report["artifact_bytes"] = {
        name: os.path.getsize(os.path.join(directory, "export", name))
        for name in ("forward.onnx", "forward.data", "adapter.npz", "interface.json")
    }
    if mode == "joint":
        for name in ("loss_gradient.onnx", "loss_gradient.data"):
            report["artifact_bytes"][name] = os.path.getsize(
                os.path.join(directory, "export", name)
            )
    if model.quantization is not None:
        report["artifact_bytes"]["quantization.json"] = os.path.getsize(
            os.path.join(directory, "export", "quantization.json")
        )
    report["forward_nodes"] = len(model.forward_graph.graph)
    report["adapter_parameter_bytes"] = sum(value.nbytes for value in final_params.values())

    # Keep completed training/export evidence even if an optional audit raises.
    def persist():
        path = os.path.join(directory, "report.json")
        with open(path + ".tmp", "w") as stream:
            json.dump(report, stream, indent=2)
        os.replace(path + ".tmp", path)

    report["requested_audits"] = {
        "reference": reference_audit,
        "strict_reference": reference_audit and not diagnose_rounding,
        "peft": peft_roundtrip,
        "fp64": fp64_audit,
        "stable_reference": stable_reference,
    }
    persist()
    if not report["frozen_buffers_unchanged"]:
        raise AssertionError("Frozen model buffers changed during training")
    if peft_roundtrip:
        from examples.huggingface.peft_adapter import roundtrip

        for row in model.mapping:
            parent, name = row["module"].rsplit(".", 1)
            module = reference.get_submodule(parent)
            setattr(module, name, getattr(module, name).base)
        report["peft"] = roundtrip(
            model,
            reference,
            sample,
            os.path.join(directory, "peft_roundtrip"),
            diagnose_rounding=diagnose_rounding,
            stable_reference=stable_reference,
        )
        with torch.no_grad():
            for name, parameter in zip(model.onnx.names, model.onnx.adapters, strict=True):
                parameter.copy_(torch.from_numpy(final_params[name]))
    if fp64_audit:
        from examples.huggingface.precision import audit64

        if mode == "joint":
            model.onnx.close()
        else:
            model.onnx.forward_session = model.onnx.backward_session = None
            model.onnx.runtime_directory.cleanup()
        report["fp64"] = audit64(model, reference, sample, [initial_params, final_params])
    persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="qwen_sft")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument(
        "--train-file", help="UTF-8 training text, one sequence per nonempty line"
    )
    parser.add_argument("--validation-file", help="UTF-8 held-out text file")
    parser.add_argument("--mode", choices=["full", "chunked", "joint"], default="full")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--reference-audit",
        action="store_true",
        help="Compare reference logits, loss and gradients at training endpoints",
    )
    reference_checks = parser.add_mutually_exclusive_group()
    reference_checks.add_argument(
        "--strict-reference",
        dest="diagnose_rounding",
        action="store_false",
        help="Require pointwise reference comparisons to pass",
    )
    reference_checks.add_argument(
        "--diagnose-rounding",
        dest="diagnose_rounding",
        action="store_true",
        help="Record pointwise reference comparisons",
    )
    parser.set_defaults(diagnose_rounding=None)
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--fp64-audit", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--projection-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--peft-roundtrip", action="store_true")
    parser.add_argument("--provider", choices=["cpu", "mlx", "reference"], default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--segment-nodes", type=int)
    parser.add_argument(
        "--stable-reference",
        action="store_true",
        help="Record extra stable FP32 CE/head reference comparisons",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                real=args.real,
                steps=args.steps,
                resume=args.resume,
                train_file=args.train_file,
                validation_file=args.validation_file,
                mode=args.mode,
                chunk_size=args.chunk_size,
                max_length=args.max_length,
                reference_audit=args.reference_audit,
                diagnose_rounding=args.diagnose_rounding,
                save_activations=args.save_activations,
                fp64_audit=args.fp64_audit,
                nf4=args.nf4,
                peft_roundtrip=args.peft_roundtrip,
                provider=args.provider,
                library=args.library,
                segment_nodes=args.segment_nodes,
                stable_reference=args.stable_reference,
                projection_precision=args.projection_precision,
            ),
            indent=2,
        )
    )
