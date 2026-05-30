import pytest
import torch

from kernel import (
    matmul_int8,
    matmul_quantize_int8,
    quantize_int8,
    quantize_int8_perrow,
)


def _assert_close_quantized(triton_out: torch.Tensor, torch_out: torch.Tensor) -> None:
    diff = (torch_out.float() - triton_out.float()).abs()
    rel = diff / torch_out.float().abs().clamp(min=1e-3)
    assert rel.mean() < 0.18, f"mean rel error {rel.mean().item():.4f}"
    assert torch.quantile(rel, 0.95) < 0.22, f"p95 rel error {torch.quantile(rel, 0.95).item():.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantize_int8_perrow():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    int_a, scale_a = quantize_int8_perrow(a)
    ref_scale = a.abs().amax(dim=1) / 127.0
    ref_int = (a / ref_scale.unsqueeze(1)).to(torch.int8)
    torch.testing.assert_close(scale_a, ref_scale, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(int_a.float(), ref_int.float(), rtol=0.0, atol=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_matmul_quantize_int8():
    torch.manual_seed(42)
    M, K, N = 128, 256, 128
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    int_b, scale_b = quantize_int8(b, axis=0)

    triton_out = matmul_quantize_int8(a, int_b, scale_b)
    torch_out = torch.matmul(a, b)
    _assert_close_quantized(triton_out, torch_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_matmul_int8_pipeline():
    torch.manual_seed(7)
    M, K, N = 64, 128, 64
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    int_a, scale_a = quantize_int8_perrow(a)
    int_b, scale_b = quantize_int8(b, axis=0)

    triton_out = matmul_int8(int_a, scale_a, int_b, scale_b)
    torch_out = torch.matmul(a, b)
    _assert_close_quantized(triton_out, torch_out)
