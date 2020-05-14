#include "gemm.hpp"

#include "kernel.hpp"

namespace
{

std::shared_ptr<Element[]> deviceAllocPitch(uint32_t rows, uint32_t cols, size_t& pitch);

}  // namespace


std::shared_ptr<Element[]> hostAlloc(uint32_t rows, uint32_t cols)
{
    Element* tmp;

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&tmp), sizeof(Element) * rows * cols, cudaHostAllocDefault));

    constexpr auto deleter = [](Element ptr[]) {
        CUDA_CHECK(cudaFreeHost(ptr));
    };

    return {tmp, deleter};
}

DeviceData allocateDeviceData(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols)
{
    DeviceData result;

    uint32_t lhs_rows_quot = lhs_rows / Devices, lhs_rows_remain = lhs_rows % Devices;
    for (auto device{0}; device < Devices; device++) {
        CUDA_CHECK(cudaSetDevice(device));
        auto& problem = result.at(device);

        problem.lhs.rows = lhs_rows_quot + (lhs_rows_remain > static_cast<uint32_t>(device));
        problem.lhs.cols = lhs_cols;

        problem.rhs.rows = problem.lhs.cols;
        problem.rhs.cols = rhs_cols;

        problem.result.rows = problem.lhs.rows;
        problem.result.cols = problem.rhs.cols;

        problem.lhs.data = deviceAllocPitch(problem.lhs.rows, problem.lhs.cols, problem.lhs.pitch);
        problem.rhs.data = deviceAllocPitch(problem.rhs.rows, problem.rhs.cols, problem.rhs.pitch);
        problem.result.data = deviceAllocPitch(problem.result.rows, problem.result.cols, problem.result.pitch);
    }

    return result;
}

void cudaGemm(DeviceData& data, std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result)
{
    std::array<cudaStream_t, Devices> streams;

    size_t begin_row{0};
    for (auto device{0}; device < Devices; device++) {
        auto& problem = data.at(device);
        auto& stream = streams.at(device);

        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&streams.at(device)));

        CUDA_CHECK(cudaMemset2DAsync(
            problem.result.data.get(), sizeof(Element) * problem.result.pitch,
            0,
            sizeof(Element) * problem.result.cols, problem.result.rows,
            stream));

        CUDA_CHECK(cudaMemcpy2DAsync(
            problem.lhs.data.get(), sizeof(Element) * problem.lhs.pitch,
            lhs.get() + begin_row * problem.lhs.cols, sizeof(Element) * problem.lhs.cols,
            sizeof(Element) * problem.lhs.cols, problem.lhs.rows,
            cudaMemcpyDefault,
            stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            problem.rhs.data.get(), sizeof(Element) * problem.rhs.pitch,
            rhs.get(), sizeof(Element) * problem.rhs.cols,
            sizeof(Element) * problem.rhs.cols, problem.rhs.rows,
            cudaMemcpyDefault,
            stream));


        kernel(
            problem.lhs.data.get(), problem.rhs.data.get(), problem.result.data.get(),
            problem.lhs.rows, problem.lhs.cols, problem.rhs.cols,
            problem.lhs.pitch, problem.rhs.pitch, problem.result.pitch,
            1, 1, 0, stream);


        CUDA_CHECK(cudaMemcpy2DAsync(
            result.get() + begin_row * problem.result.cols, sizeof(Element) * problem.result.cols,
            problem.result.data.get(), sizeof(Element) * problem.rhs.pitch,
            sizeof(Element) * problem.result.cols, problem.result.rows,
            cudaMemcpyDefault,
            stream));

        begin_row += problem.lhs.rows;
    }

    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

namespace
{

std::shared_ptr<Element[]> deviceAllocPitch(uint32_t rows, uint32_t cols, size_t& pitch)
{
    Element* tmp;

    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&tmp), &pitch, sizeof(Element) * rows, cols));
    pitch /= sizeof(Element);  // convert pitch in bytes to one in elems

    constexpr auto deleter = [](Element ptr[]) {
        CUDA_CHECK(cudaFree(ptr));
    };

    return {tmp, deleter};
}

}  // namespace
