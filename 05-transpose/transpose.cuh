#pragma once

#include <cuda_helpers.h>
#include <cuda_runtime.h>

void transpose_cuda(const float *in, float *out, int rows, int cols);
