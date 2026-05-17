import torch
import triton
import triton.language as tl

from autotune_config import get_autotune_config

INT8_MAX = 127.0


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
    ],
    key=["K"],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr,
    a_ptr,
    as_ptr,
    M,
    K,
    stride_fpam,
    stride_fpak,
    stride_am,
    stride_ak,
    stride_asm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pass


def quantize_int8_perrow(fpa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass


def quantize_int8(weight: torch.Tensor, axis: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
    reset_to_zero=["c_ptr"],
)
@triton.jit
def perrow_w8a8_matmul_kernel(
    a_ptr,
    as_ptr,
    b_ptr,
    bs_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_bk,
    stride_bn,
    stride_bsn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pass

def matmul_int8(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    pass


def matmul_quantize_int8(
    fpa: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    pass
