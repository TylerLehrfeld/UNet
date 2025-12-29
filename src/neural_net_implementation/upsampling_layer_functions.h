#ifndef UPSAMPLING_UNET_FUNCS
#define UPSAMPLING_UNET_FUNCS

#include "BN_functions.h"
#include "activation_functions.h"
#include "cuda_lib.h"
#include "layer.h"

inline void Layer::forward_upsample(float *input,
                                    int batch_size,
                                    bool use_relu_activation,
                                    bool use_batch_norm,
                                    bool inference) {
  const int scale = kernel_size;
  int out_image_size = scale * scale * in_width * in_height;
  int num_activations = batch_size * out_image_size * out_channels;
  cudaUpsample(input,
               parameters,
               activations_int_1,
               scale,
               in_channels,
               out_channels,
               in_height,
               in_width,
               batch_size);

  cudaCheckErr();
  // float gpu_val = cudaValToCPU(activations, 0);
  // std::cout << "[DEBUG] upsample check - GPU value: " << gpu_val <<
  // std::endl;

  if(inference) {
    forward_batch_norm_inference(activations_int_1,
                                 activations_int_2,
                                 parameters,
                                 BN_stats,
                                 out_channels,
                                 out_height,
                                 out_width);

    cudaCheckErr();

    // gpu_val = cudaValToCPU(activations, 0);
    // std::cout << "[DEBUG] upsample check - GPU value: " << gpu_val <<
    // std::endl;
  } else {
    forward_batch_norm_training(activations_int_1,
                                activations_int_2,
                                parameters,
                                BN_stats,
                                BN_batch_stats,
                                out_channels,
                                out_height,
                                out_width,
                                batch_size);

    cudaCheckErr();

    // gpu_val = cudaValToCPU(activations, 0);
    // std::cout << "[DEBUG] upsample check - GPU value: " << gpu_val <<
    // std::endl;
  }
  forward_relu(activations_int_2, activations, num_activations);

  cudaCheckErr();

  // gpu_val = cudaValToCPU(activations, 0);
  // std::cout << "[DEBUG] upsample check - GPU value: " << gpu_val <<
  // std::endl;
}

inline void Layer::backward_upsample(float *grad_activations,
                                     float *input,
                                     int batch_size) {
  const int scale = kernel_size;
  int out_image_size = scale * scale * in_width * in_height;
  int num_activations = batch_size * out_image_size * out_channels;

  // Backward through ReLU
  cudaReluBackward(grad_activations,
                   activations_int_2,
                   grad_activations_int_2,
                   num_activations);
  if(cudaHasNans(grad_activations_int_2,
                 batch_size * (shape_size(output_shape)))) {
    std::cout << "backward upsample relu HAS NANS" << std::endl;
  };
  cudaCheckErr();
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

  cudaCheckErr();
  if(cudaHasNans(grad_activations_int_1,
                 batch_size * (shape_size(output_shape)))) {
    std::cout << "backward batch norm BN HAS NANS" << std::endl;
  };
  // Backward through upsample
  cudaUpsampleBackward(grad_activations_int_1,
                       input,
                       parameters,
                       dLdX,
                       dLdW,
                       scale,
                       in_channels,
                       out_channels,
                       in_height,
                       in_width,
                       batch_size);

  cudaCheckErr();
  if(cudaHasNans(dLdX, batch_size * (shape_size(input_shape)))) {
    std::cout << "backward upsample HAS NANS" << std::endl;
  };
}

#endif
