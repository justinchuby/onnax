# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Read whole-process macOS physical-footprint counters outside timed regions."""

import ctypes
import os
import sys


class _RUsageInfoV4(ctypes.Structure):
    # sys/resource.h: rusage_info_v4.
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        ("ri_" + name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
            "cpu_time_qos_default",
            "cpu_time_qos_maintenance",
            "cpu_time_qos_background",
            "cpu_time_qos_utility",
            "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive",
            "billed_system_time",
            "serviced_system_time",
            "logical_writes",
            "lifetime_max_phys_footprint",
            "instructions",
            "cycles",
            "billed_energy",
            "serviced_energy",
            "interval_max_phys_footprint",
            "runnable_time",
        )
    ]


def physical_footprint():
    """Return current and lifetime peak process footprint in bytes on macOS."""
    if sys.platform != "darwin":
        return {"available": False, "reason": "macOS proc_pid_rusage is required"}
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    usage = _RUsageInfoV4()
    if function(os.getpid(), 4, ctypes.byref(usage)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return {
        "available": True,
        "current_bytes": int(usage.ri_phys_footprint),
        "lifetime_peak_bytes": int(usage.ri_lifetime_max_phys_footprint),
        "current_resident_bytes": int(usage.ri_resident_size),
        "scope": "macOS kernel whole-process physical footprint; includes setup and runtime",
    }
