"""One fused unpack/scale kernel; GEMM itself continues to use PyTorch FP16."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack(codes, scales, output, size: tl.constexpr, group_size: tl.constexpr,
            BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(codes + offset // 2, mask=offset < size, other=0).to(tl.int32)
    value = ((packed >> ((offset % 2) * 4)) & 15) - 8
    scale = tl.load(scales + offset // group_size, mask=offset < size, other=0).to(tl.float32)
    tl.store(output + offset, value.to(tl.float32) * scale, mask=offset < size)


def unpack_int4(codes, scales, size, group_size, dtype=torch.float16):
    output = torch.empty(size, device=codes.device, dtype=dtype)
    _unpack[(triton.cdiv(size, 1024),)](codes, scales, output, size, group_size, BLOCK=1024)
    return output
