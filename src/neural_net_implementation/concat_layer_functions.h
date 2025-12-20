#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_concat(float *activations_1,
                                  float *activations_2,
                                  int batch_size) {
  cuda_concat(activations_1,
              activations_2,
              activations,
              batch_size,
              input_shape[2] * input_shape[3],
              input_shape[0],
              input_shape[1]);

  cuda_check_err();
  // float gpu_val = cudaValToCPU(activations, 0);
  // std::cout << "[DEBUG] concat check - GPU value: " << gpu_val << std::endl;
  if(has_nans(activations, batch_size * shape_size(output_shape))) {
    std::cout << "HAS NANS" << std::endl;
  };
}

inline void Layer::backward_concat(float *dLdY,
                                   float *inputs_1,
                                   float *inputs_2,
                                   int batch_size) {
  int C1 = input_shape[0];
  int C2 = input_shape[1];
  cuda_concat_backward(dLdY, dLdX, in_height, in_width, C1, C2, batch_size);

  cuda_check_err();
  if(has_nans(dLdX,
              batch_size * (shape_size(parents[0]->output_shape) +
                            shape_size(parents[1]->output_shape)))) {
  }
}
