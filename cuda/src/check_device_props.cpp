#include "check_device_props.hpp"

#include "error.hpp"


void checkDeviceProps()
{
    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));

    if (devices < 2) {
        throw UnsupportedDevice("less than 2 devices available");
    }

    for (auto device{0}; device < devices; ++device) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (!prop.unifiedAddressing) {
            throw UnsupportedDevice(std::string{"device["} + std::to_string(device) + "] (named \"" + prop.name + "\")"
                                    + " doesn't support unified addressing");
        }
    }
}
