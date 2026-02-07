#include "cuda_helpers.h"

#include <stdexcept>
#include <string>

void CheckStatus(const cudaError_t& status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
    }
}
