#pragma once

#include "cuda_manager.h"

inline CudaProcessID
CudaManager::addProcess(FunctionID cuda_function_id,
                        const std::vector<PointerID> &pointer_parameter_ids,
                        const std::vector<int> &integer_parameters,
                        const std::vector<CudaProcessID> &parent_processes) {
  
  
  return -1;
}
