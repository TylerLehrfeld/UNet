#pragma once

#include "cuda_manager_lib.h"
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using PointerID = int;
using CudaProcessID = int;
using StreamID = int;

enum class FunctionID {
  kConvolve,
  kMaxPool,
};

enum class PointerLocation {
  kDevice,
  kHost,
  kStack,
};

enum class PointerType {
  kFloat,
  kDouble,
  kInt,
  kHalf,
};

// Inclusive range (range of length 5 would have start 0 and end 4)
struct Range {
  Range(size_t start, size_t end) : start(start), end(end) {}
  size_t start;
  size_t end;
} __attribute__((packed));

class CudaMemoryManager {
  std::byte *device_heap_;
  size_t num_bytes_;
  std::vector<Range> range_list_;

  CudaMemoryManager(std::byte *device_heap, size_t num_bytes)
      : device_heap_(device_heap), num_bytes_(num_bytes) {}
  void addRange(size_t num_bytes, PointerID ID);
};

struct ManagedCudaPointer {
  void *pointer;
  PointerLocation pointer_location;
  PointerType type;
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
  DeviceMemInfo current_device_info_;
  std::unordered_map<PointerID, ManagedCudaPointer> active_pointers_;
  std::unordered_map<CudaProcessID, StreamID> active_streams_;
  std::byte *device_heap_;

  CudaManager(size_t desired_memory_required_bytes = 0) {
    checkLibReturn(resetDevice());
    checkLibReturn(getDeviceMemInfo(current_device_info_));
    checkLibReturn(getPointer(device_heap_, desired_memory_required_bytes));
  }
  CudaProcessID addProcess(FunctionID cuda_function_id,
                           const std::vector<PointerID> &pointer_parameter_ids,
                           const std::vector<int> &integer_parameters,
                           const std::vector<CudaProcessID> &parent_processes);

  PointerID cudaGetPointer(PointerType, int length);

  void globalSync();

private:
};
