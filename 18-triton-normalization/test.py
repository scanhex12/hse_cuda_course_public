import pytest
import torch

from kernel import layernorm_relu_fp16


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton kernel",
)
@pytest.mark.parametrize("B,N", [
    (1, 64),
    (2, 128),
    (4, 256),
    (8, 1024),
])
def test_layernorm_relu_fp16(B, N):
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-5

    X = torch.randn(B, N, device=device, dtype=torch.float16)
    gamma = torch.randn(N, device=device, dtype=torch.float16)
    beta = torch.randn(N, device=device, dtype=torch.float16)

    ref = torch.nn.functional.layer_norm(
        X, (N,), gamma, beta, eps
    )
    ref = torch.relu(ref)

    out = layernorm_relu_fp16(X, gamma, beta, eps=eps)

    assert out.dtype == torch.float16
    assert out.shape == (B, N)

    torch.testing.assert_close(
        out,
        ref,
        rtol=1e-3,
        atol=1e-3,
    )
