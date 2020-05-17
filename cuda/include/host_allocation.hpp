#pragma once

#include <cstdint>
#include <memory>


namespace CudaMM
{

template <uint32_t Rows = 0, uint32_t Cols = 0>
std::shared_ptr<Element[]> hostAlloc(uint32_t rows = Rows, uint32_t cols = Cols);

}  // namespace CudaMM

#include "host_allocation.ipp"
