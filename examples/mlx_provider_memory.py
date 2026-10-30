# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Inspect the MLX allocator bundled with the native ONNX Runtime provider."""

import ctypes
import os


class ProviderMemory:
    """Use the provider's bundled MLX C ABI at completed execution boundaries.

    Call after synchronous runtime execution and external array synchronization.
    Cache operations affect the provider allocator shared by sessions in this process.
    Query Python MLX separately through mlx.core.
    """

    def __init__(self):
        import onnxruntime_ep_mlx

        self.path = os.path.join(
            os.path.dirname(onnxruntime_ep_mlx.library_path()), "libmlxc.dylib"
        )
        self.library = ctypes.CDLL(self.path)
        for name in ("active", "cache", "peak"):
            function = getattr(self.library, f"mlx_get_{name}_memory")
            function.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
            function.restype = ctypes.c_int
        self.library.mlx_clear_cache.argtypes = []
        self.library.mlx_clear_cache.restype = ctypes.c_int

    def read(self):
        result = {}
        for name in ("active", "cache", "peak"):
            value = ctypes.c_size_t()
            if getattr(self.library, f"mlx_get_{name}_memory")(ctypes.byref(value)) != 0:
                raise RuntimeError(f"Provider MLX {name} memory query failed")
            result[name + "_bytes"] = value.value
        return result

    def clear_cache(self):
        if self.library.mlx_clear_cache() != 0:
            raise RuntimeError("Provider MLX cache clear failed")
