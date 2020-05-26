#include "kernel.hpp"

#include "gemm.hpp"
#include "size.hpp"


namespace CudaMM
{

namespace Kernel
{

Matrix2D<0, 0>::Matrix2D(CudaMM::Matrix2D<0, 0> const& mat)
    : pitch{mat.pitch}, rows{mat.rows}, cols{mat.cols}
{
}


template <>
__global__
__launch_bounds__(BlockSize, MinBlocksPerMultiprocessor)
void gemm_impl<LhsRows / Devices, LhsCols, RhsCols>(
    ProblemData<LhsRows / Devices, LhsCols, RhsCols> problem,
    const Element* __restrict__ lhs, const Element* __restrict__ rhs, Element* __restrict__ result)
{
    extern __shared__ Element shared_memory[];
    // Element lhs_shared[LhsSharedRows][LhsSharedCols];
    // Element rhs_shared[RhsSharedRows][RhsSharedCols];
    // Element result_shared[ResultSharedRows][ResultSharedCols];
    Element(*lhs_shared)[LhsSharedCols] = reinterpret_cast<Element(*)[LhsSharedCols]>(shared_memory);
    Element(*rhs_shared)[RhsSharedCols] = reinterpret_cast<Element(*)[RhsSharedCols]>(lhs_shared + LhsSharedRows);
    Element(*result_shared)[ResultSharedCols] = reinterpret_cast<Element(*)[ResultSharedCols]>(rhs_shared + RhsSharedRows);

    Element result_reg[ThreadRows][ThreadCols] = {};


    auto const block_offset_row = BlockRows * blockIdx.y,
               block_offset_col = BlockCols * blockIdx.x;
    auto const &row_in_block = threadIdx.y,
               &col_in_block = threadIdx.x;
    auto const thread_idx = BlockColThreads * row_in_block + col_in_block;


    // storing to lhs_shared is divided into (LhsLoadUnitRows x LhsLoadUnitCols (= BlockSize))
    constexpr uint32_t LhsLoadUnitCols = gcd(Stride, BlockSize),
                       LhsLoadUnitRows = BlockSize / LhsLoadUnitCols;
    // storing to rhs_shared is divided into (RhsLoadUnitRows x RhsLoadUnitCols (= BlockSize))
    constexpr uint32_t RhsLoadUnitCols = BlockColThreads * gcd(ThreadCols, BlockRowThreads),
                       RhsLoadUnitRows = BlockSize / RhsLoadUnitCols;

    auto const lhs_load_row = thread_idx / LhsLoadUnitCols,
               lhs_load_col = thread_idx % LhsLoadUnitCols;
    auto const rhs_load_row = thread_idx / RhsLoadUnitCols,
               rhs_load_col = thread_idx % RhsLoadUnitCols;


    lhs += (block_offset_row + lhs_load_row * (BlockRows / LhsLoadUnitRows)) * problem.lhs.pitch + blockIdx.z * MaxAccumulation + lhs_load_col;
    rhs += (blockIdx.z * MaxAccumulation + rhs_load_row * (Stride / RhsLoadUnitRows)) * problem.rhs.pitch + block_offset_col + rhs_load_col;


    // initialize result_shared
    {
#pragma unroll
        for (uint32_t col_begin{0}; col_begin < (BlockColThreads * ThreadSharedCols); col_begin += BlockColThreads) {
#pragma unroll
            for (uint32_t row_begin{0}; row_begin < (BlockRowThreads * ThreadSharedRows); row_begin += BlockRowThreads) {

                result_shared[row_begin + row_in_block][col_begin + col_in_block] = 0;
            }
        }
    }


#pragma unroll
    for (uint32_t lhs_stride_col{0}; lhs_stride_col < MaxAccumulation; lhs_stride_col += Stride) {
        auto const& rhs_stride_row = lhs_stride_col;

        if (lhs_stride_col != 0) {
            __syncthreads();
        }

        // load lhs
        {
#pragma unroll
            for (uint32_t offset_col{0}; offset_col < Stride; offset_col += LhsLoadUnitCols) {
#pragma unroll
                for (uint32_t offset_row{0}; offset_row < (BlockRows / LhsLoadUnitRows); offset_row++) {
                    lhs_shared[offset_row + lhs_load_row * (BlockRows / LhsLoadUnitRows)][offset_col + lhs_load_col]
                        = lhs[offset_row * problem.lhs.pitch + (lhs_stride_col + offset_col)];
                }
            }
        }

        // load rhs
        {
#pragma unroll
            for (uint32_t offset_col{0}; offset_col < BlockCols; offset_col += RhsLoadUnitCols) {
#pragma unroll
                for (uint32_t offset_row{0}; offset_row < (Stride / RhsLoadUnitRows); offset_row++) {
                    rhs_shared[offset_row + rhs_load_row * (Stride / RhsLoadUnitRows)][offset_col + rhs_load_col]
                        = rhs[(rhs_stride_row + offset_row) * problem.rhs.pitch + offset_col];
                }
            }
        }

        __syncthreads();


#pragma unroll
        for (uint32_t in_stride{0}; in_stride < Stride; in_stride++) {
            if (ThreadRows < ThreadCols) {
                // rows is less than cols, so load lhs first
                Element lhs_cache[ThreadRows];
#pragma unroll
                for (uint32_t row{0}; row < ThreadRows; row++) {
                    lhs_cache[row] = lhs_shared[row * BlockRowThreads + row_in_block][in_stride];
                }

                // then, load each of rhs and calc
#pragma unroll
                for (uint32_t col{0}; col < ThreadCols; col++) {
                    auto rhs_cache = rhs_shared[in_stride][col * BlockColThreads + col_in_block];

#pragma unroll
                    for (uint32_t row{0}; row < ThreadRows; row++) {
                        auto const result = lhs_cache[row] * rhs_cache;

                        if (row < ThreadSharedRows) {
                            if (col < ThreadSharedCols) {
                                result_shared[row * BlockRowThreads + row_in_block][col * BlockColThreads + col_in_block] += result;

                            } else {
                                result_reg[row][col] += result;
                            }

                        } else {
                            result_reg[row][col] += result;
                        }
                    }
                }

            } else {
                // cols is less than rows, so load rhs first
                Element rhs_cache[ThreadCols];
#pragma unroll
                for (uint32_t col{0}; col < ThreadCols; col++) {
                    rhs_cache[col] = rhs_shared[in_stride][col * BlockColThreads + col_in_block];
                }

                // then, load each of lhs and calc
#pragma unroll
                for (uint32_t row{0}; row < ThreadRows; row++) {
                    auto lhs_cache = lhs_shared[row * BlockRowThreads + row_in_block][in_stride];

#pragma unroll
                    for (uint32_t col{0}; col < ThreadCols; col++) {
                        auto const result = lhs_cache * rhs_cache[col];

                        if (row < ThreadSharedRows) {
                            if (col < ThreadSharedCols) {
                                result_shared[row * BlockRowThreads + row_in_block][col * BlockColThreads + col_in_block] += result;

                            } else {
                                result_reg[row][col] += result;
                            }

                        } else {
                            result_reg[row][col] += result;
                        }
                    }
                }
            }
        }
    }


#pragma unroll
    for (uint32_t col{0}; col < ThreadCols; col++) {
#pragma unroll
        for (uint32_t row{0}; row < ThreadRows; row++) {
            if (row < ThreadSharedRows) {
                if (col < ThreadSharedCols) {
                    atomicAdd(&result[(block_offset_row + row_in_block + row * BlockRowThreads) * problem.result.pitch + block_offset_col + col_in_block + col * BlockColThreads],
                        result_shared[row * BlockRowThreads + row_in_block][col * BlockColThreads + col_in_block]);
                } else {
                    atomicAdd(&result[(block_offset_row + row_in_block + row * BlockRowThreads) * problem.result.pitch + block_offset_col + col_in_block + col * BlockColThreads],
                        result_reg[row][col]);
                }

            } else {
                atomicAdd(&result[(block_offset_row + row_in_block + row * BlockRowThreads) * problem.result.pitch + block_offset_col + col_in_block + col * BlockColThreads],
                    result_reg[row][col]);
            }
        }
    }
}


template <>
void gemm<Devices, 0, LhsRows / Devices, LhsCols, RhsCols>(
    CudaMM::ProblemData<LhsRows / Devices, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    // for 128-byte transactions on loading from global memory
    static_assert(BlockCols % WarpThreads == 0, "BlockCols % WarpThreads == 0");
    static_assert(Stride % WarpThreads == 0, "Stride % WarpThreads == 0");

    // loading from global memory should be unrolled loops
    static_assert((ThreadRows * Stride) % BlockColThreads == 0, "(ThreadRows * Stride) % BlockColThreads == 0");
    static_assert((ThreadCols * Stride) % BlockRowThreads == 0, "(ThreadCols * Stride) % BlockRowThreads == 0");

    // block size should be a multiple of warp size
    static_assert(BlockSize % WarpThreads == 0, "BlockSize % WarpThreads == 0");

    // result stored in shared memory should be a part
    static_assert(ThreadSharedRows <= ThreadRows, "ThreadSharedRows <= ThreadRows");
    static_assert(ThreadSharedCols <= ThreadCols, "ThreadSharedCols <= ThreadCols");
    static_assert(ThreadSharedRows != 0 || ThreadSharedCols == 0, "ThreadSharedRows != 0 || ThreadSharedCols == 0");
    static_assert(ThreadSharedRows == 0 || ThreadSharedCols != 0, "ThreadSharedRows == 0 || ThreadSharedCols != 0");

    // restriction of hardware
    static_assert(BlockSize <= 1024, "BlockSize <= 1024");
    static_assert(MinBlocksPerMultiprocessor <= 32, "MinBlocksPerMultiprocessor <= 32");
    static_assert(BlockSize * MinBlocksPerMultiprocessor <= 2048, "BlockSize * MinBlocksPerMultiprocessor <= 2048");
    static_assert(sizeof(Element) * SharedSize * MinBlocksPerMultiprocessor <= 0x18000, "sizeof(Element) * SharedSize * MinBlocksPerMultiprocessor <= 0x18000");

    static_assert(problem.result.rows % BlockRows == 0, "problem.result.rows % BlockRows == 0");
    static_assert(problem.result.cols % BlockCols == 0, "problem.result.cols % BlockCols == 0");
    static_assert(problem.lhs.cols % MaxAccumulation == 0, "problem.lhs.cols % MaxAccumulation == 0");
    static_assert(MaxAccumulation % Stride == 0, "MaxAccumulation % Stride == 0");


    CUDA_CHECK(cudaMemset2DAsync(
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        0,
        sizeof(Element) * problem.result.cols, problem.result.rows,
        0));

    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.lhs.data.get(), sizeof(Element) * problem.lhs.pitch,
        lhs, sizeof(Element) * problem.lhs.cols,
        sizeof(Element) * problem.lhs.cols, problem.lhs.rows,
        cudaMemcpyDefault,
        0));
    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.rhs.data.get(), sizeof(Element) * problem.rhs.pitch,
        rhs, sizeof(Element) * problem.rhs.cols,
        sizeof(Element) * problem.rhs.cols, problem.rhs.rows,
        cudaMemcpyDefault,
        0));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_impl<LhsRows / Devices, LhsCols, RhsCols>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Element) * SharedSize));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_impl<LhsRows / Devices, LhsCols, RhsCols>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    gemm_impl<<<{problem.result.cols / BlockCols,
                    problem.result.rows / BlockRows,
                    problem.lhs.cols / MaxAccumulation},
        {BlockColThreads, BlockRowThreads},
        sizeof(Element) * SharedSize>>>(

        toKernel(problem),
        problem.lhs.data.get(), problem.rhs.data.get(), problem.result.data.get());


    CUDA_CHECK(cudaMemcpy2DAsync(
        result, sizeof(Element) * problem.result.cols,
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        sizeof(Element) * problem.result.cols, problem.result.rows,
        cudaMemcpyDefault,
        0));
}

template <>
void gemm<Devices, 1, LhsRows / Devices, LhsCols, RhsCols>(
    CudaMM::ProblemData<LhsRows / Devices, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    gemm<Devices, 0, LhsRows / Devices, LhsCols, RhsCols>(
        problem,
        lhs, rhs, result);
}

}  // namespace Kernel

}  // namespace CudaMM

template void CudaMM::gemm<LhsRows, LhsCols, RhsCols>(
    CudaMM::DeviceData<LhsRows, LhsCols, RhsCols>&,
    const Element* lhs, const Element* rhs, Element* result);
