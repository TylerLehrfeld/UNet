#include "cuda_manager_lib.h"

CudaManagerLibReturnValue getDeviceMemInfo(DeviceMemInfo &ret_info) {
  ret_info.used_bytes = 0;
  return check(cudaMemGetInfo(&ret_info.free_bytes, &ret_info.total_bytes));
}

CudaManagerLibReturnValue getPointer(void *dPointer, size_t bytes) {
  return check(cudaMalloc(&dPointer, bytes));
}

CudaManagerLibReturnValue resetDevice() { return check(cudaDeviceReset()); }

CudaManagerLibReturnValue check(cudaError_t err) {
  if(err == cudaSuccess) {
    return CudaManagerLibReturnValue::kSuccess;
  } else {
    return CudaManagerLibReturnValue::kFailure;
  }
}
