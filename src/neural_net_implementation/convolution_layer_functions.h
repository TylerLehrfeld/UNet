#include "BN_functions.h"
#include "activation_functions.h"
#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_convolution(float *input,
                                       int batch_size,
                                       bool use_relu_activation,
                                       bool use_batch_norm,
                                       bool inference) {
  // we access input as input[batch_num][channel][h][w]
  // similarly we calculate output/activations as
  // activations[batch_num][channel][h][w] Both arrays are flattened
  int num_activations = batch_size * out_channels * out_height * out_width;

  cudaConvolve(parameters,
               input,
               activations_int_1,
               num_activations,
               batch_size,
               kernel_size,
               in_channels,
               out_channels,
               out_height,
               out_width,
               in_height,
               in_width,
               padding,
               stride);
  cuda_check_err();

  // int debug_batch = 0;
  // int debug_channel = 0;
  // int debug_h = 9;
  // int debug_w = 9;

  // float cpu_sum = 0.0f;

  // int input_i_center = debug_h * stride - padding + (kernel_size - 1) / 2;
  // int input_j_center = debug_w * stride - padding + (kernel_size - 1) / 2;

  // for(int c = 0; c < in_channels; ++c) {
  //   for(int kh = -kernel_size / 2; kh <= kernel_size / 2; ++kh) {
  //     for(int kw = -kernel_size / 2; kw <= kernel_size / 2; ++kw) {
  //       int h_in = input_i_center + kh;
  //       int w_in = input_j_center + kw;

  //      if(h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
  //        int input_idx = debug_batch * in_channels * in_height * in_width +
  //                        c * in_height * in_width + h_in * in_width + w_in;

  //        int weight_idx = c * out_channels * kernel_size * kernel_size +
  //                         debug_channel * kernel_size * kernel_size +
  //                         (kh + kernel_size / 2) * kernel_size +
  //                         (kw + kernel_size / 2);

  //        float input_val = cudaValToCPU(input, input_idx);
  //        float param_val = cudaValToCPU(parameters, weight_idx);

  //        cpu_sum += input_val * param_val;
  //      }
  //    }
  //  }

  //  // add bias for this output channel
  //  cpu_sum += cudaValToCPU(
  //      parameters,
  //      in_channels * out_channels * kernel_size * kernel_size +
  //      debug_channel);
  //}

  //// GPU value
  // int act_idx = debug_batch * out_channels * out_height * out_width +
  //               debug_channel * out_height * out_width + debug_h * out_width
  //               + debug_w;

  // float gpu_val = cudaValToCPU(activations_int_1, act_idx);
  // std::cout << "[DEBUG] Convolution check - GPU value: " << gpu_val
  //           << " | CPU recompute: " << cpu_sum << std::endl;
  if(inference) {
    forward_batch_norm_inference(activations_int_1,
                                 activations_int_2,
                                 BN_parameters,
                                 BN_stats,
                                 out_channels,
                                 out_height,
                                 out_width);

    cuda_check_err();
  } else {
    forward_batch_norm_training(activations_int_1,
                                activations_int_2,
                                BN_parameters,
                                BN_stats,
                                BN_batch_stats,
                                out_channels,
                                out_height,
                                out_width,
                                batch_size);
    cuda_check_err();
  }
  // gpu_val = cudaValToCPU(activations_int_2, act_idx);
  // std::cout << "[DEBUG] Convolution check - GPU value: " << gpu_val
  //           << std::endl;

  if(activation_type == SIGMOID) {
    forward_sigmoid(activations_int_2, activations, num_activations);
    cuda_check_err();
  } else {
    forward_relu(activations_int_2, activations, num_activations);
    cuda_check_err();
  }
  // gpu_val = cudaValToCPU(activations, act_idx);
  // std::cout << "[DEBUG] Convolution check - GPU value: " << gpu_val
  //           << std::endl;
  if(has_nans(activations, batch_size * (shape_size(output_shape)))) {
    std::cout << "HAS NANS" << std::endl;
  };
}

inline void Layer::backward_convolution(float *grad_activations,
                                        float *input,
                                        int batch_size) {

  int num_activations = batch_size * out_channels * out_height * out_width;

  // Backward through activation (ReLU or Sigmoid)
  if(activation_type == SIGMOID) {
    cuda_sigmoid_backward(
        grad_activations, activations, grad_activations_int_2, num_activations);
  } else {
    cuda_relu_backward(grad_activations,
                       activations_int_2,
                       grad_activations_int_2,
                       num_activations);
  }
  if(has_nans(grad_activations_int_2,
              batch_size * (shape_size(output_shape)))) {
    std::cout << "backward convolution relu HAS NANS" << std::endl;
  };
  // Backward through batch norm
  cuda_BN_backward(grad_activations_int_2,
                   activations_int_1,
                   BN_batch_stats,
                   BN_parameters,
                   grad_activations_int_1,
                   dLdW + num_weights_and_biases,
                   out_channels,
                   out_height,
                   out_width,
                   batch_size);

  if(has_nans(grad_activations_int_1,
              batch_size * (shape_size(output_shape)))) {
    std::cout << "backward convolution BN HAS NANS" << std::endl;
  };
  // Backward through convolution
  cudaConvolveBackward(grad_activations_int_1,
                       input,
                       parameters,
                       dLdX,
                       dLdW,
                       batch_size,
                       kernel_size,
                       in_channels,
                       out_channels,
                       out_height,
                       out_width,
                       in_height,
                       in_width,
                       padding,
                       stride);
  if(has_nans(dLdX, batch_size * (shape_size(input_shape)))) {
    std::cout << "backward convolution HAS NANS" << std::endl;
  };
}
