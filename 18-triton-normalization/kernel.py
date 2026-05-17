import triton
import triton.language as tl
import torch


@triton.jit
def layernorm_relu_fp16_kernel(
    X_ptr,
    Gamma_ptr,
    Beta_ptr,
    Z_ptr,
    stride_x,
    stride_z,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: Implement the kernel


def layernorm_relu_fp16(X, gamma, beta, eps=1e-5):
    assert X.dtype == torch.float16
    assert gamma.dtype == torch.float16
    assert beta.dtype == torch.float16
    assert X.is_cuda

    # TODO: Implement the kernel
