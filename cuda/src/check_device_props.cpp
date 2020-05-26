#include "check_device_props.hpp"

#include "error.hpp"
#include "gemm.hpp"


namespace CudaMM
{

void checkDeviceProps()
{
    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));

    if (devices < Devices) {
        throw UnsupportedDevice("less than 2 devices available");
    }

    for (auto device{0}; device < Devices; ++device) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (!prop.unifiedAddressing) {
            throw UnsupportedDevice(std::string{"device["} + std::to_string(device) + "] (named \"" + prop.name + "\")"
                                    + " doesn't support unified addressing");
        }

        if (prop.asyncEngineCount < 2) {
            throw UnsupportedDevice(std::string{"device["} + std::to_string(device) + "] (named \"" + prop.name + "\")"
                                    + " has less than 2 asyncronous engines");
        }
    }
}

}  // namespace CudaMM
