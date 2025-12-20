#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_max_pool(float *input, int batch_size) {
  // int debug_batch = 0;
  // int debug_channel = 0;
  // int debug_h = 9;
  // int debug_w = 9;

  // int act_idx = debug_batch * out_channels * out_height * out_width +
  //               debug_channel * out_height * out_width + debug_h * out_width
  //               + debug_w;
  // float gpu_val = cudaValToCPU(input, act_idx);
  // std::cout << "[DEBUG] max pool check - GPU value: " << gpu_val <<
  // std::endl; gpu_val = cudaValToCPU(input, act_idx + 1); std::cout <<
  // "[DEBUG] max pool check - GPU value: " << gpu_val << std::endl; gpu_val =
  // cudaValToCPU(input, act_idx + in_width); std::cout << "[DEBUG] max pool
  // check - GPU value: " << gpu_val << std::endl; gpu_val = cudaValToCPU(input,
  // act_idx + in_width); std::cout << "[DEBUG] max pool check - GPU value: " <<
  // gpu_val << std::endl;
  cudaMaxPool(input,
              activations,
              batch_size,
              stride,
              kernel_size,
              in_channels,
              in_height,
              in_width);

  cuda_check_err();
  // gpu_val = cudaValToCPU(activations, 0);
  // std::cout << "[DEBUG] max pool check - GPU value: " << gpu_val <<
  // std::endl;
  if(has_nans(activations, batch_size * (shape_size(output_shape)))) {
    std::cout << "HAS NANS" << std::endl;
  };
}

inline void
Layer::backward_max_pool(float *dLdY, float *inputs, int batch_size) {
  cudaMaxPoolBackward(dLdY,
                      inputs,
                      activations,
                      dLdX,
                      batch_size,
                      stride,
                      in_channels,
                      in_height,
                      in_width);

  cuda_check_err();
  if(has_nans(dLdX, batch_size * (shape_size(parents[0]->output_shape)))) {
    std::cout << "HAS NANS" << std::endl;
  };
}
