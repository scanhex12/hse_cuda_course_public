import triton
import triton.language as tl
import torch

@triton.jit
def rgb_to_gray_kernel(
    img_ptr,
    out_ptr,
    height,
    width,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr
):
    # TODO: Implement the kernel

def rgb2grey(img):
    H, W, C = img.shape
    out = torch.empty_like(img)
    
    grid = (H, W)
    rgb_to_gray_kernel[grid](
        img, out,
        H, W,
        img.stride(0), img.stride(1),
        BLOCK_SIZE=1
    )
    return out
