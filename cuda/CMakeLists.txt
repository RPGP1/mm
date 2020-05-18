cmake_minimum_required(VERSION 3.12)


project(MM_CUDA CXX CUDA)

file(GLOB_RECURSE MM_CUDA_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cu")
add_executable(mm_cuda "${MM_CUDA_SOURCES}")
target_include_directories(mm_cuda PRIVATE
    "include")
target_include_directories(mm_cuda SYSTEM PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>)
target_link_libraries(mm_cuda PRIVATE
    mm_tester)


set_target_properties(mm_cuda PROPERTIES
    CUDA_STANDARD_REQUIRED ON
    CUDA_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    LINKER_LANGUAGE CUDA)


# flags for executable file
target_compile_options(mm_cuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Og>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:RelWithDebgInfo>:-O3>
    $<$<CONFIG:RelWithDebgInfo>:-g>
    )
# flags for compile errors
target_compile_options(mm_cuda PRIVATE
    "-ftemplate-backtrace-limit=0"
    $<$<COMPILE_LANGUAGE:CXX>:-fmax-errors=10>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler="-fmax-errors=10">
    )
# flags for warnings
set(WARNING_FLAGS
    all
    extra
    conversion
    switch-default
    disabled-optimization
    format
    init-self
    overloaded-virtual
    float-equal
    no-old-style-cast
    no-pragmas
    no-ignored-attributes
    return-type
    )
foreach(FLAG IN LISTS WARNING_FLAGS)
    target_compile_options(mm_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-W${FLAG}>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler="-W${FLAG}">
        )
endforeach()
