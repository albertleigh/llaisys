import os
import sys
import ctypes
from pathlib import Path

from .qwen2 import load_qwen2
from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    # On Windows, add the library directory to the DLL search path so that
    # dependent DLLs (e.g. openblas.dll) placed alongside llaisys.dll are found.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib_dir))
        # Add CUDA toolkit bin directory for cudart/cublas DLLs
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if os.path.isdir(cuda_bin):
                os.add_dll_directory(cuda_bin)
            # CUDA 13+ places cublas DLLs in bin/x64
            cuda_bin_x64 = os.path.join(cuda_path, "bin", "x64")
            if os.path.isdir(cuda_bin_x64):
                os.add_dll_directory(cuda_bin_x64)
        # Add Intel oneAPI directories for MKL / Intel OpenMP runtime DLLs
        mkl_root = os.environ.get(
            "MKLROOT",
            r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest",
        )
        mkl_bin = os.path.join(mkl_root, "bin")
        if os.path.isdir(mkl_bin):
            os.add_dll_directory(mkl_bin)
        cmplr_root = os.environ.get(
            "CMPLR_ROOT",
            r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest",
        )
        cmplr_bin = os.path.join(cmplr_root, "bin")
        if os.path.isdir(cmplr_bin):
            os.add_dll_directory(cmplr_bin)

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)

__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
]
