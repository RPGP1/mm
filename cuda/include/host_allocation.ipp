#pragma once

#include "host_allocation.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>


namespace CudaMM
{

template <uint32_t Rows, uint32_t Cols>
std::shared_ptr<Element[]> hostAlloc(uint32_t rows, uint32_t cols)
{
    Element* tmp;

    auto constexpr Size = sizeof(Element) * Rows * Cols;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&tmp),
        (Size != 0 ? Size : sizeof(Element) * rows * cols),
        cudaHostAllocDefault));

    constexpr auto deleter = [](Element ptr[]) {
        CUDA_CHECK(cudaFreeHost(ptr));
    };

    return {tmp, deleter};
}

}  // namespace CudaMM
