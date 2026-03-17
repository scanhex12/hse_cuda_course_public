#include <cuda_runtime.h>

void choose_top_moe_experts(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k);
    