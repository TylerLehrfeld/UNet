#pragma once

#include "cuda_manager_lib.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using PointerID = int;
using CudaProcessID = int;
using StreamID = int;

enum class pointerLocation {
  kDevice,
  kHost,
  kStack,
};

enum class pointerType {
  kFloat,
  kDouble,
  kInt,
  kHalf,
};

struct ManagedCudaPointer {
  void *pointer;
  pointerLocation pointer_location;
  pointerType type;
};

inline void checkLibReturn(CudaManagerLibReturnValue val,
                           const char *file = __FILE__,
                           int line = __LINE__) {

  if(val == CudaManagerLibReturnValue::kFailure) {
    std::stringstream ss;
    ss << "Failure in file: " << std::string(file) << " on line " << std::endl;
    throw std::runtime_error(ss.str());
  }
}

class CudaManager {
public:
  DeviceMemInfo current_device_info;
  std::unordered_map<PointerID, ManagedCudaPointer> active_pointers_;
  std::unordered_map<CudaProcessID, StreamID> active_streams_;
  CudaManager(size_t desired_memory_required_bytes = 0) {
    checkLibReturn(resetDevice());
    checkLibReturn(getDeviceMemInfo(current_device_info));
  }
  CudaProcessID addProcess(int cuda_function_id,
                           const std::vector<PointerID> &pointer_parameter_ids,
                           const std::vector<int> &integer_parameters,
                           const std::vector<CudaProcessID> &parent_processes);

  PointerID getCudaPointer(pointerType, int length);

  void globalSync();

private:
};
