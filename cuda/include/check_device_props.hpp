#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>


struct UnsupportedDevice : public std::runtime_error {
    explicit UnsupportedDevice(std::string const& message)
        : std::runtime_error{message}
    {
    }
};

void checkDeviceProps();
