#pragma once

#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>


namespace CudaMM
{

struct CudaError : public std::runtime_error {
    explicit CudaError(cudaError_t const& err, std::string const& file, size_t line);
};

void cudaCheckError(cudaError_t const& err, std::string const& file, size_t line);

#define CUDA_CHECK(arg) cudaCheckError((arg), __FILE__, __LINE__)

}  // namespace CudaMM
