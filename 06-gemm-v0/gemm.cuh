#pragma once

#include <cuda_helpers.h>
#include <cuda_runtime.h>

void gemm_cuda(float alpha, const float *A, const float *B, float beta,
               float *C, int M, int K, int N);
