#pragma once

#include <cuda_runtime.h>

void solve(const float* input, float gamma, float beta, float* output, int N,
           float eps);
