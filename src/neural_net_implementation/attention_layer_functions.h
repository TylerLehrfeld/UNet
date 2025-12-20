#include "cuda_lib.h"
#include "layer.h"

inline void
Layer::forward_attention(float *input_x, float *input_g, int batch_size) {

  cuda_attention(activations_int_1,
                 activations_int_2,
                 activations,
                 parameters,
                 input_x,
                 input_g,
                 batch_size,
                 in_height,
                 in_width,
                 in_channels,
                 input_shape[3],
                 in_channels / 2);

  cuda_check_err();

  //float gpu_val = cudaValToCPU(activations, 0);
  //std::cout << "[DEBUG] attention check - GPU value: " << gpu_val << std::endl;
  if(has_nans(activations, batch_size * shape_size(output_shape))) {
    std::cout << "HAS NANS" << std::endl;
  };
}

inline void Layer::backward_attention(float *grad_activations,
                                      float *input_x,
                                      float *input_g,
                                      int batch_size) {
  int input_x_size = batch_size * shape_size(parents[0]->output_shape);
  cuda_attention_backward(grad_activations,
                          activations_int_1,
                          activations_int_2,
                          input_x,
                          input_g,
                          parameters,
                          grad_activations_int_1,
                          grad_activations_int_2,
                          dLdX,
                          dLdX + input_x_size,
                          dLdW,
                          batch_size,
                          in_height,
                          in_width,
                          in_channels,
                          input_shape[3],
                          in_channels / 2);
  cuda_check_err();
  if(has_nans(dLdX,
              batch_size * (shape_size(parents[0]->output_shape) +
                            shape_size(parents[1]->output_shape)))) {
    std::cout << "HAS NANS" << std::endl;
  };
}
