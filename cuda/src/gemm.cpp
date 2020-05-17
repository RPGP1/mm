#include "gemm.hpp"


namespace CudaMM
{

Matrix2D<0, 0>::Matrix2D(uint32_t rows, uint32_t cols)
    : pitch{Cols}, rows{rows}, cols{cols}
{
}

}  // namespace CudaMM
