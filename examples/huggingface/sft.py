# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mobius Qwen3 ONNX fine-tuning through TRL SFTTrainer."""

import argparse
import functools
import hashlib
import json
import os

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast, Qwen3Config
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTConfig, SFTTrainer

from examples.huggingface.bridge import OnnxModule, frozen_hashes
from examples.huggingface.qwen import (
    MODEL_ID,
    REVISION,
    build_base,
    decompose,
    inject,
    reference_model,
)
from examples.lora.train import session


def texts():
    with open(os.path.join(os.path.dirname(__file__), "corpus.txt")) as stream:
        return [line.strip() for line in stream if line.strip()]


def tokenizer(real=False):
    if real:
        result = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        result.pad_token = result.eos_token
        result.padding_side = "right"
        return result
    backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(
        texts(),
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
        if mode not in ("full", "chunked"):
            raise ValueError("Expected full or chunked training")
        self.mode = mode
        self.forward_graph = model
        if mode == "chunked":
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
        value = self.onnx(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        if self.mode == "chunked":
            if labels is not None:
                return CausalLMOutput(loss=self.head(value, labels, num_items_in_batch))
            logits = self.head.forward(value)
        else:
            logits = value
        loss = None if labels is None else token_loss(logits, labels, num_items_in_batch)
        return CausalLMOutput(loss=loss, logits=logits)


def prepare(
    real=False,
    *,
    mode="full",
    chunk_size=128,
    optimize=True,
    save_activations=False,
    nf4=False,
    session_factory=session,
):
    reference = reference_model(real)
    base = build_base(reference)
    adapted, params, mapping = inject(base)
    graph = decompose(adapted)
    quantization = None
    if nf4:
        from examples.huggingface.nf4 import lower_decoder

        graph, quantization = lower_decoder(graph, reference)
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
    return model, reference


class OnnxSFTTrainer(SFTTrainer):
    """Use the ONNX chunked loss without recomputing logits for TRL metrics."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.mode == "full":
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        from transformers import Trainer

        return Trainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)


def make_trainer(model, tok, directory, *, real=False, steps=None, callbacks=None):
    lines = texts()
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
        max_length=64,
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
        train_dataset=Dataset.from_dict({"text": lines[:-4] * 8}),
        eval_dataset=Dataset.from_dict({"text": lines[-4:]}),
        callbacks=callbacks,
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
    diagnose_rounding=False,
    save_activations=False,
    fp64_audit=False,
    nf4=False,
    peft_roundtrip=False,
    provider="cpu",
    library=None,
    segment_nodes=None,
    stable_reference=False,
):
    if provider not in ("cpu", "mlx"):
        raise ValueError("Expected cpu or mlx provider")
    if provider == "cpu" and (library is not None or segment_nodes is not None):
        raise ValueError("Library and segmentation options require the mlx provider")
    runtime = session
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
    torch.set_num_threads(4 if real else 1)
    model, reference = prepare(
        real,
        mode=mode,
        chunk_size=chunk_size,
        optimize=optimize,
        save_activations=save_activations,
        nf4=nf4,
        session_factory=runtime,
    )
    initial_frozen = frozen_hashes(model.forward_graph)
    tok = tokenizer(real)
    from examples.huggingface.validation import add_reference_adapters, audit

    reference = add_reference_adapters(reference, model)
    sample = tok(texts()[0], return_tensors="pt")
    sample = {k: v for k, v in sample.items() if k in ("input_ids", "attention_mask")}
    sample["labels"] = sample["input_ids"].clone()
    initial_params = model.onnx.arrays()
    initial_audit = audit(model, reference, sample, diagnose_rounding=diagnose_rounding)

    def stable_control():
        from examples.huggingface.reference_head import pairwise_head
        from examples.huggingface.reference_losses import stable_token_loss

        with pairwise_head(reference):
            return audit(
                model,
                reference,
                sample,
                diagnose_rounding=True,
                reference_loss=stable_token_loss,
            )

    initial_stable = stable_control() if stable_reference else None
    trainer = make_trainer(model, tok, directory, real=real, steps=steps)
    before = trainer.evaluate()["eval_loss"]
    trainer.train(resume_from_checkpoint=resume)
    after = trainer.evaluate()["eval_loss"]
    final_audit = audit(model, reference, sample, diagnose_rounding=diagnose_rounding)
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
        "validation_loss": [before, after],
        "audits": [initial_audit, final_audit],
        "quantization": model.quantization,
        "provider": provider,
        "segment_target_nodes": segment_nodes,
        "ort_graph_optimization": "disabled",
        "frozen_buffers_unchanged": frozen_hashes(model.forward_graph) == initial_frozen,
    }
    if provider == "mlx":
        report["cpu_fallback_disabled"] = True
        report["library"] = os.path.abspath(library) if library else "installed plugin"
        if library:
            with open(library, "rb") as stream:
                report["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if stable_reference:
        report["stable_reference_audits"] = {
            "loss": "target-excluded stable CE",
            "head_backward": "FP32 pairwise K blocks of 128",
            "checks": [initial_stable, final_stable],
        }
    final_params = model.onnx.arrays()
    report["artifact_bytes"] = {
        name: os.path.getsize(os.path.join(directory, "export", name))
        for name in ("forward.onnx", "forward.data", "adapter.npz", "interface.json")
    }
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
        )
        with torch.no_grad():
            for name, parameter in zip(model.onnx.names, model.onnx.adapters, strict=True):
                parameter.copy_(torch.from_numpy(final_params[name]))
    if fp64_audit:
        from examples.huggingface.precision import audit64

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
    parser.add_argument("--mode", choices=["full", "chunked"], default="full")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--diagnose-rounding", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--fp64-audit", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--peft-roundtrip", action="store_true")
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
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
                mode=args.mode,
                chunk_size=args.chunk_size,
                diagnose_rounding=args.diagnose_rounding,
                save_activations=args.save_activations,
                fp64_audit=args.fp64_audit,
                nf4=args.nf4,
                peft_roundtrip=args.peft_roundtrip,
                provider=args.provider,
                library=args.library,
                segment_nodes=args.segment_nodes,
                stable_reference=args.stable_reference,
            ),
            indent=2,
        )
    )
