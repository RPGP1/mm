#include "error.hpp"


namespace CudaMM
{

CudaError::CudaError(cudaError_t const& err, std::string const& file, size_t line)
    : std::runtime_error{std::string{"cudaError "} + std::to_string(err) + " occurred. "
                         + "( " + file + ":" + std::to_string(line) + " ) "
                         + "ref https://docs.nvidia.com/cuda/archive/9.1/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"}
{
}

void cudaCheckError(cudaError_t const& err, std::string const& file, size_t line)
{
    if (err != cudaSuccess) {
        throw CudaError{err, file, line};
    }
}

}  // namespace CudaMM
