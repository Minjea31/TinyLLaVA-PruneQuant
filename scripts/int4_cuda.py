"""Native s4*s4->s32 CUDA WMMA kernels, launched on the active PyTorch stream.

The existing nvcc builds a device-only cubin. Loading it through libcuda avoids
linking a second CUDA runtime or requiring a matching PyTorch C++ extension ABI.
"""

import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from filelock import FileLock
import torch

from common import ROOT


_MODULES = {}
_DRIVER = None
WARPS = 4


def _driver():
    global _DRIVER
    if _DRIVER is None:
        driver = ct.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ct.c_uint]
        driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
        driver.cuModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
        driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
        driver.cuGetErrorString.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
        driver.cuLaunchKernel.argtypes = [ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p,
                                         ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
        _DRIVER = driver
        _check(driver.cuInit(0), "cuInit")
    return _DRIVER


def _check(result, operation):
    if result:
        message = ct.c_char_p()
        if _DRIVER:
            _DRIVER.cuGetErrorString(result, ct.byref(message))
        raise RuntimeError(f"{operation}: CUDA driver error {result}: {message.value!r}")


def _module(device):
    with torch.cuda.device(device):
        torch.cuda.current_stream()
        driver = _driver()
        context = ct.c_void_p()
        _check(driver.cuCtxGetCurrent(ct.byref(context)), "cuCtxGetCurrent")
        if not context.value:
            raise RuntimeError("A current PyTorch CUDA context is required")
        if context.value in _MODULES:
            return _MODULES[context.value]
        capability = torch.cuda.get_device_capability()
        if capability[0] not in (7, 8) or capability < (7, 5):
            raise RuntimeError(f"This WMMA s4 backend supports SM75/SM8x; got {capability}")
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc is required to build the native INT4 cubin")
        source = ROOT / "kernels/int4_gemm.cu"
        compiler = subprocess.check_output([nvcc, "--version"], text=True)
        architecture = f"sm_{capability[0]}{capability[1]}"
        key = hashlib.sha256(source.read_bytes() + compiler.encode() + architecture.encode()).hexdigest()[:20]
        directory = ROOT / "model/.cache/int4_cuda" / key
        directory.mkdir(parents=True, exist_ok=True)
        cubin = directory / "int4_gemm.cubin"
        with FileLock(str(directory / "build.lock")):
            if not cubin.exists():
                temporary = directory / "int4_gemm.tmp.cubin"
                command = [nvcc, "--cubin", "-O3", "--fmad=false", "-std=c++14",
                           f"-arch={architecture}", str(source), "-o", str(temporary)]
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (directory / "build.log").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"INT4 CUDA build failed; see {directory / 'build.log'}\n{result.stdout[-3000:]}")
                temporary.replace(cubin)
                metadata = {"architecture": architecture, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "compiler": compiler, "build_command": command}
                (directory / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
        module = ct.c_void_p()
        _check(driver.cuModuleLoad(ct.byref(module), str(cubin).encode()), "cuModuleLoad")
        functions = {}
        for name in ("repack_weight", "quantize_activation", "raw_int4_gemm", "scaled_int4_gemm"):
            function = ct.c_void_p()
            _check(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()), "cuModuleGetFunction")
            functions[name] = function
        result = {"module": module, "functions": functions, "cubin": cubin, "architecture": architecture,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        _MODULES[context.value] = result
        return result


def launch(name, device, blocks, arguments):
    with torch.cuda.device(device):
        module = _module(device)
        holders = [ct.c_void_p(value.data_ptr()) if isinstance(value, torch.Tensor)
                   else ct.c_void_p(0) if value is None else ct.c_int(value) for value in arguments]
        parameters = (ct.c_void_p * len(holders))(
            *(ct.cast(ct.byref(value), ct.c_void_p) for value in holders))
        stream = torch.cuda.current_stream(device)
        _check(_driver().cuLaunchKernel(module["functions"][name], blocks, 1, 1,
                                       WARPS * 32, 1, 1, 0, ct.c_void_p(stream.cuda_stream),
                                       parameters, None), name)


def kernel_info(device="cuda", include_instructions=False):
    module = _module(torch.device(device))
    result = {key: str(value) for key, value in module.items() if key in ("cubin", "architecture", "source_sha256")}
    result.update(backend="cuda_wmma_s4", multiply="signed_int4", accumulator="int32",
                  dequantized_operand_gemm=False, performance_tuned=False)
    if include_instructions:
        executable = shutil.which("cuobjdump")
        if not executable:
            raise RuntimeError("cuobjdump is required to verify generated INT4 instructions")
        sass = subprocess.check_output([executable, "--dump-sass", str(module["cubin"])], text=True)
        sections = re.split(r"Function\s*:\s*(\w+)", sass)
        per_kernel = {}
        for i in range(1, len(sections), 2):
            per_kernel[sections[i]] = sorted(set(line.strip() for line in sections[i+1].splitlines() if "IMMA" in line))
        for name in ("raw_int4_gemm", "scaled_int4_gemm"):
            if not any("S4.S4" in line for line in per_kernel.get(name, [])):
                raise RuntimeError(f"{name} does not contain signed INT4 IMMA instructions")
        instructions = sorted(set(line for values in per_kernel.values() for line in values))
        result["imma_instructions"] = instructions
        result["imma_by_kernel"] = {name: per_kernel[name] for name in ("raw_int4_gemm", "scaled_int4_gemm")}
    return result


def repack_weight(codes, n, k, group, physical_group):
    physical_k = k // group * physical_group
    output = torch.empty(((n + 7) // 8 * 8, physical_k // 2), dtype=torch.uint8, device=codes.device)
    launch("repack_weight", codes.device, min(65535, (output.numel() + 127) // 128),
           [codes, output, n, k, group, physical_group])
    return output


def quantize_activation(inputs, group, physical_group):
    if inputs.dtype != torch.float16 or not inputs.is_cuda or not inputs.is_contiguous() or inputs.ndim != 2:
        raise ValueError("INT4 activation quantization requires a contiguous CUDA FP16 matrix")
    m, k = inputs.shape
    padded_m = (m + 7) // 8 * 8
    packed = torch.empty((padded_m, k // group * physical_group // 2), dtype=torch.uint8, device=inputs.device)
    scales = torch.empty((padded_m, k // group), dtype=torch.float16, device=inputs.device)
    launch("quantize_activation", inputs.device, (padded_m + WARPS - 1) // WARPS,
           [inputs, packed, scales, m, k, group, physical_group])
    return packed, scales


def raw_gemm(a, b, m, n):
    if (a.dtype != torch.uint8 or b.dtype != torch.uint8 or a.ndim != 2 or b.ndim != 2
            or a.device != b.device or not a.is_cuda or a.shape[1] != b.shape[1]
            or not a.is_contiguous() or not b.is_contiguous()
            or a.shape[0] < (m + 7) // 8 * 8 or b.shape[0] < (n + 7) // 8 * 8
            or a.shape[1] % 16 or m < 1 or n < 1):
        raise ValueError("Packed signed INT4 operands need matching padded CUDA layouts")
    output = torch.empty((m, n), dtype=torch.int32, device=a.device)
    tiles = ((m + 7) // 8) * ((n + 7) // 8)
    launch("raw_int4_gemm", a.device, (tiles + WARPS - 1) // WARPS,
           [a, b, output, m, n, a.shape[1] * 2])
    return output


def scaled_gemm(a, b, a_scales, w_scales, bias, m, n, k, weight_group, group, physical_group):
    output = torch.empty((m, n), dtype=torch.float16, device=a.device)
    tiles = ((m + 7) // 8) * ((n + 7) // 8)
    launch("scaled_int4_gemm", a.device, (tiles + WARPS - 1) // WARPS,
           [a, b, a_scales, w_scales, bias, output, m, n, k, weight_group, group, physical_group])
    return output
