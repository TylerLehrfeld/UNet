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
  cudaCheckErr();

  if(inference) {
    forward_batch_norm_inference(activations_int_1,
                                 activations_int_2,
                                 BN_parameters,
                                 BN_stats,
                                 out_channels,
                                 out_height,
                                 out_width);

    cudaCheckErr();
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

    cudaCheckErr();
  }
  // gpu_val = cudaValToCPU(activations_int_2, act_idx);
  // std::cout << "[DEBUG] Convolution check - GPU value: " << gpu_val
  //           << std::endl;

  if(activation_type == SIGMOID) {
    forward_sigmoid(activations_int_2, activations, num_activations);
    cudaCheckErr();
  } else {
    forward_relu(activations_int_2, activations, num_activations);
    cudaCheckErr();
  }
  // gpu_val = cudaValToCPU(activations, act_idx);
  // std::cout << "[DEBUG] Convolution check - GPU value: " << gpu_val
  //           << std::endl;
  if(cudaHasNans(activations, batch_size * (shape_size(output_shape)))) {
    std::cout << "HAS NANS" << std::endl;
  };
}

inline void Layer::backward_convolution(float *grad_activations,
                                        float *input,
                                        int batch_size) {

  int num_activations = batch_size * out_channels * out_height * out_width;

  // Backward through activation (ReLU or Sigmoid)
  if(activation_type == SIGMOID) {
    cudaSigmoidBackward(
        grad_activations, activations, grad_activations_int_2, num_activations);

  } else {
    cudaReluBackward(grad_activations,
                     activations_int_2,
                     grad_activations_int_2,
                     num_activations);
  }
  if(cudaHasNans(grad_activations_int_2,
                 batch_size * (shape_size(output_shape)))) {
    std::cout << "backward convolution relu HAS NANS" << std::endl;
  };
  // Backward through batch norm
  cudaBatchNormBackward(grad_activations_int_2,
                        activations_int_1,
                        BN_batch_stats,
                        BN_parameters,
                        grad_activations_int_1,
                        dLdW + num_weights_and_biases,
                        out_channels,
                        out_height,
                        out_width,
                        batch_size);

  if(cudaHasNans(grad_activations_int_1,
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
  if(cudaHasNans(dLdX, batch_size * (shape_size(input_shape)))) {
    std::cout << "backward convolution HAS NANS" << std::endl;
  };
}
