#pragma once

#include "definition.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>


void kernel(
    Element* lhs, Element* rhs, Element* result,
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    size_t lhs_pitch, size_t rhs_pitch, size_t result_pitch,
    dim3 blocks, dim3 threads, size_t shared_memory_size = 0, cudaStream_t stream = 0);
