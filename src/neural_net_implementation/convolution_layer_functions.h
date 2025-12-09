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

  std::cout << "paraneter val before conv: " << cudaValToCPU(parameters, 0)
            << std::endl;
  convolve(parameters,
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
  std::cout << "paraneter val after conv: " << cudaValToCPU(parameters, 0)
            << std::endl;

  // Insert a basic check here to show that the convolution is actually working

  // --- DEBUG CHECK: verify one output value ---
  int debug_batch = 0;
  int debug_channel = 0;
  int debug_h = 10;
  int debug_w = 10;

  float cpu_sum = 0.0f;
  for(int c = 0; c < in_channels; ++c) {
    for(int kh = 0; kh < kernel_size; ++kh) {
      for(int kw = 0; kw < kernel_size; ++kw) {
        int h_in = debug_h * stride - padding + kh - kernel_size / 2;
        int w_in = debug_w * stride - padding + kw - kernel_size / 2;
        if(h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
          int input_idx = debug_batch * in_channels * in_height * in_width +
                          c * in_height * in_width + h_in * in_width + w_in;
          int weight_idx = c * out_channels * kernel_size * kernel_size +
                           debug_channel * kernel_size * kernel_size +
                           kh * kernel_size + kw;

          float input_val = cudaValToCPU(input, input_idx);
          float param_val = cudaValToCPU(parameters, weight_idx);
          cpu_sum += input_val * param_val;
        }
      }
    }
    // add bias for this output channel
    cpu_sum += cudaValToCPU(
        parameters,
        in_channels * out_channels * kernel_size * kernel_size + debug_channel);
  }

  // copy GPU value back to host
  float gpu_val;
  int act_idx = debug_batch * out_channels * out_height * out_width +
                debug_channel * out_height * out_width + debug_h * out_width +
                debug_w;
  gpu_val = cudaValToCPU(activations_int_1, act_idx);
  std::cout << "[DEBUG] Convolution check - GPU value: " << gpu_val
            << " | CPU recompute: " << cpu_sum << std::endl;

  // --- continue as usual ---
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
  if(activation_type == SIGMOID) {
    forward_sigmoid(activations_int_2, activations, num_activations);
    cuda_check_err();
  } else {
    forward_relu(activations_int_2, activations, num_activations);
    cuda_check_err();
  }
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

  // Backward through convolution
  convolve_backward(grad_activations_int_1,
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
}
