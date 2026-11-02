# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Pinned native Transformers sources for example-layer ONNX training tasks."""

import re
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer


@dataclass(frozen=True)
class PretrainedSource:
    """A checkpoint and tokenizer revision, with explicit parameter-prefix mapping."""

    model_id: str
    revision: str
    prefix_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not self.model_id or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Pretrained sources require a model ID and full commit SHA")
        if any(not old or not new for old, new in self.prefix_map):
            raise ValueError("Checkpoint prefixes must be nonempty")

    def checkpoint_name(self, reference_name):
        """Map a native text-model parameter back to its original checkpoint name."""
        for checkpoint_prefix, reference_prefix in self.prefix_map:
            if reference_name.startswith(reference_prefix):
                return checkpoint_prefix + reference_name[len(reference_prefix) :]
        return reference_name


def load_reference(config_type, model_type, source, *, cache_dir=None):
    """Load FP32 weights completely; fail before training on any loading mismatch."""
    options = dict(revision=source.revision, cache_dir=cache_dir)
    config = config_type.from_pretrained(source.model_id, **options)
    config._attn_implementation = "eager"
    model, loading = model_type.from_pretrained(
        source.model_id,
        config=config,
        dtype=torch.float32,
        key_mapping={"^" + re.escape(old): new for old, new in source.prefix_map} or None,
        output_loading_info=True,
        **options,
    )
    failures = {key: sorted(map(str, values)) for key, values in loading.items() if values}
    if failures:
        raise ValueError(f"Pretrained weight loading was incomplete: {failures}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("The pretrained qualification path requires FP32 parameters")
    model.config._name_or_path = source.model_id
    model.config._commit_hash = source.revision
    model.config.onnax_pretrained = {
        "version": 1,
        "model_id": source.model_id,
        "revision": source.revision,
        "reference_class": type(model).__name__,
        "checkpoint_to_reference_prefix": dict(source.prefix_map),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dtype": "float32",
        "loading": {key: [] for key in loading},
    }
    return model.eval()


def load_tokenizer(source, *, cache_dir=None):
    """Use the checkpoint's tokenizer with explicit right padding."""
    tokenizer = AutoTokenizer.from_pretrained(
        source.model_id,
        revision=source.revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
