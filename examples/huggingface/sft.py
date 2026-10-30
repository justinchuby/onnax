# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mobius Qwen3 ONNX fine-tuning through TRL SFTTrainer."""

import argparse
import json
import os

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast, Qwen3Config
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTConfig, SFTTrainer

from examples.huggingface.bridge import OnnxModule
from examples.huggingface.qwen import (
    MODEL_ID,
    REVISION,
    build_base,
    decompose,
    inject,
    reference_model,
)


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

    def __init__(self, config, model, params, mapping):
        super().__init__(config)
        self.loss_type = "ForCausalLM"
        self.onnx = OnnxModule(model, params, output="logits")
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
        logits = self.onnx(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        loss = None if labels is None else token_loss(logits, labels, num_items_in_batch)
        return CausalLMOutput(loss=loss, logits=logits)


def prepare(real=False):
    reference = reference_model(real)
    base = build_base(reference)
    adapted, params, mapping = inject(base)
    graph = decompose(adapted)
    return OnnxQwen3(reference.config, graph, params, mapping), reference


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
    return SFTTrainer(
        model=model,
        args=args,
        processing_class=tok,
        train_dataset=Dataset.from_dict({"text": lines[:-4] * 8}),
        eval_dataset=Dataset.from_dict({"text": lines[-4:]}),
        callbacks=callbacks,
    )


def run(directory, *, real=False, steps=None, resume=None):
    torch.set_num_threads(4 if real else 1)
    model, reference = prepare(real)
    tok = tokenizer(real)
    from examples.huggingface.validation import add_reference_adapters, audit

    reference = add_reference_adapters(reference, model)
    sample = tok(texts()[0], return_tensors="pt")
    sample = {k: v for k, v in sample.items() if k in ("input_ids", "attention_mask")}
    sample["labels"] = sample["input_ids"].clone()
    initial_audit = audit(model, reference, sample)
    trainer = make_trainer(model, tok, directory, real=real, steps=steps)
    before = trainer.evaluate()["eval_loss"]
    trainer.train(resume_from_checkpoint=resume)
    after = trainer.evaluate()["eval_loss"]
    final_audit = audit(model, reference, sample)
    model.onnx.export(os.path.join(directory, "export"))
    tok.save_pretrained(os.path.join(directory, "export"))
    report = {
        "model": MODEL_ID if real else "tiny-qwen3",
        "revision": REVISION if real else None,
        "steps": trainer.state.global_step,
        "validation_loss": [before, after],
        "audits": [initial_audit, final_audit],
    }
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="qwen_sft")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output_dir, real=args.real, steps=args.steps, resume=args.resume),
            indent=2,
        )
    )
