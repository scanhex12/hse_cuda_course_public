#pragma once

// Q and V: N x d, K: d x N
void flash_attention(const float* Q, const float* K, const float* V, float* O, int N, int d);
