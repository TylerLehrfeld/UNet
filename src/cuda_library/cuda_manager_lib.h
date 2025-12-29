#pragma once

#include <cstddef>
#include <cuda_runtime.h>

enum class CudaManagerLibReturnValue { kSuccess, kFailure };

struct DeviceMemInfo {
  size_t free_bytes;
  size_t total_bytes;
  size_t used_bytes;
};

CudaManagerLibReturnValue resetDevice();

CudaManagerLibReturnValue getDeviceMemInfo(DeviceMemInfo &ret_info);

CudaManagerLibReturnValue getPointer(void *dPointer, size_t bytes);

CudaManagerLibReturnValue check(cudaError_t err);

