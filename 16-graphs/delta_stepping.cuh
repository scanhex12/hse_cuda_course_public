#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <limits>

namespace delta_stepping {

inline void shortest_paths(int num_vertices, const int* csr_row_ptr, const int* csr_col_idx,
                           const int* edge_weight, int source, int delta, int* distances_out) {
}

}  // namespace delta_stepping
