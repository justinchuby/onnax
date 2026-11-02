# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Content-addressed immutable tensor payloads for co-located ONNX modules."""

import hashlib
import os
import tempfile

import numpy as np
import onnx
import onnx_ir as ir


class SharedWeights:
    """Write immutable tensor payloads once for several co-located ONNX graphs."""

    def __init__(self, directory, *, data_name="weights.data"):
        self.directory = os.fspath(directory)
        if os.path.basename(data_name) != data_name:
            raise ValueError("Shared data must use a filename in the model directory")
        self.stem, self.extension = os.path.splitext(data_name)
        self.stream = tempfile.NamedTemporaryFile(  # ruff: ignore[open-file-with-context-handler] - Owned by this context manager.
            dir=directory, prefix=self.stem + "-", suffix=".tmp", mode="wb", delete=False
        )
        self.data_name = os.path.basename(self.stream.name)
        self.digest = hashlib.sha256()
        self.paths = []
        self.payloads = {}
        self.tensors = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        if args[0] is not None:
            return
        # Immutable content-addressed payloads also make re-exporting a loaded
        # module safe: existing memory mappings are never truncated or changed.
        final_name = self.stem + "-" + self.digest.hexdigest() + self.extension
        final_path = os.path.join(self.directory, final_name)
        if os.path.exists(final_path):
            with open(final_path, "rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != self.digest.hexdigest()
                ):
                    raise ValueError("Existing shared weight payload has the wrong digest")
            os.unlink(self.stream.name)
        else:
            os.replace(self.stream.name, final_path)
        for path in self.paths:
            model = onnx.load(path, load_external_data=False)
            for tensor in model.graph.initializer:
                for entry in tensor.external_data:
                    if entry.key == "location" and entry.value == self.data_name:
                        entry.value = final_name
            onnx.save(model, path)
        self.data_name = final_name

    def save(self, model, name):
        result = model.clone()
        for key, value in result.graph.initializers.items():
            tensor = value.const_value
            if tensor.nbytes < 64 * 1024:
                continue
            cached = self.tensors.get(id(tensor))
            if cached is not None and cached[0] is tensor:
                offset, length = cached[1]
            else:
                array = np.ascontiguousarray(tensor.numpy())
                data = array.reshape(-1).view(np.uint8).data
                identity = (tensor.dtype, tuple(array.shape), hashlib.sha256(data).digest())
                if identity not in self.payloads:
                    offset = self.stream.tell()
                    self.stream.write(data)
                    self.digest.update(data)
                    self.payloads[identity] = offset, array.nbytes
                offset, length = self.payloads[identity]
                self.tensors[id(tensor)] = tensor, (offset, length)
                del array, data
            value.const_value = ir.ExternalTensor(
                self.data_name,
                offset,
                length,
                tensor.dtype,
                shape=ir.Shape(tensor.shape),
                name=key,
                base_dir=self.directory,
            )
        self.stream.flush()
        path = os.path.join(self.directory, name + ".onnx")
        ir.save(result, path)
        self.paths.append(path)
        return path
