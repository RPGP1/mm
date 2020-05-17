#include "check_device_props.hpp"
#include "definition.hpp"
#include "error.hpp"
#include "gemm.hpp"
#include "host_allocation.hpp"

#include "mm/tester.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <memory>

#include <iomanip>

#include "size.hpp"
extern template void CudaMM::gemm<LhsRows, LhsCols, RhsCols>(
    CudaMM::DeviceData<LhsRows, LhsCols, RhsCols>&,
    const Element* lhs, const Element* rhs, Element* result);


int main(int argc, char* argv[])
try {
    namespace CM = CudaMM;

    CM::checkDeviceProps();


    using Element = CM::Element;
    MM::Tester<Element> tester(argc, argv);


    auto& problem = tester.problem();
    std::cout << "# Problem: " << tester.problem_file_path() << std::endl
              << "# Size: lhs[ " << problem.lhs_rows() << " ][ " << problem.lhs_cols() << " ]"
              << " * rhs[ " << problem.rhs_rows() << " ][ " << problem.rhs_cols() << " ]" << std::endl;

    assert(LhsRows == 0 || LhsRows == problem.lhs_rows());
    assert(LhsCols == 0 || LhsCols == problem.lhs_cols());
    assert(RhsCols == 0 || RhsCols == problem.rhs_cols());


    // memory allocation
    auto lhs = CM::hostAlloc<LhsRows, LhsCols>(problem.lhs_rows(), problem.lhs_cols());
    auto rhs = CM::hostAlloc<RhsRows, RhsCols>(problem.rhs_rows(), problem.rhs_cols());
    auto result = CM::hostAlloc<ResultRows, ResultCols>(problem.lhs_rows(), problem.rhs_cols());
    problem.get(lhs.get(), rhs.get(), problem.lhs_cols(), problem.rhs_cols());


    // calculation with timer
    {
        auto device_data = CM::allocateDeviceData<LhsRows, LhsCols, RhsCols>(problem.lhs_rows(), problem.lhs_cols(), problem.rhs_cols());

        auto timer = tester.timer();

        CM::gemm(device_data, lhs.get(), rhs.get(), result.get());
    }

    {
        using millisec_double = std::chrono::duration<double, std::milli>;
        using sec_double = std::chrono::duration<double>;
        using std::chrono::duration_cast;
        auto duration = duration_cast<millisec_double>(tester.elapsed()).count();
        auto gflops = (2.0 * problem.lhs_rows() * problem.lhs_cols() * problem.rhs_cols()) / duration_cast<sec_double>(tester.elapsed()).count() / 1e+9;
        std::cout << std::setprecision(20);
        std::cout << "# Elapsed [ms]: " << duration << std::endl
                  << "# FLOPS [Gflops]: " << gflops << std::endl;
    }


    MM::Problem::Result<Element>::strict_standard_digits = 1;
    MM::Problem::Result<Element>::loose_standard_digits = 4;
    Element max_diff{0}, max_abs_answer{0};
    uint32_t max_row{0}, max_col{0};
    auto score = problem.score(result.get(), problem.rhs_cols(), [&](uint32_t row, uint32_t col, Element calced, Element answer) {
        using std::abs;
        if (abs(answer - calced) > max_diff) {
            max_row = row;
            max_col = col;
            max_diff = abs(answer - calced);
        }
        max_abs_answer = std::max(max_abs_answer, abs(answer));
        std::cerr << std::setprecision(20);
        // std::cerr << "### @( " << row << ", " << col << " ) "
        //           << calced << " != Ans = " << answer << std::endl;
    });
    std::cout << "# Standard (Strict): " << score.strict_standard << std::endl
              << "# Standard (Loose): " << score.loose_standard << std::endl
              << "# Difference count (Strict): " << score.strict_violations << std::endl
              << "# Difference count (Loose): " << score.loose_violations << std::endl
              << "# Max Wrongness: " << score.max_difference << std::endl;
    std::cout << "# @ (" << max_row << ", " << max_col << ")" << std::endl
              << "# max_abs_answer = " << max_abs_answer << std::endl;

} catch (MM::TesterImpl::CommandLineHelp const& e) {
    std::cerr << e.what() << std::endl;
    return 0;
} catch (MM::TesterImpl::BadCommandLineArguments const& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
