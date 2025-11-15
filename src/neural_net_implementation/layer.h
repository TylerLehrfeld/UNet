
#ifndef UNET_LAYER
#define UNET_LAYER

#include "cuda_lib.h"
#include <iostream>
#include <string>
#include <vector>
enum Layer_type {
  CONV_LAYER,
  BN_LAYER,
  MAX_POOL_LAYER,
  ATTENTION_LAYER,
  FC_LAYER,
  CONCAT_LAYER,
  UPSAMPLING_LAYER,
};
struct LayerDesc {
  Layer_type type;
  std::vector<int> shape_descriptor;
  std::vector<int> parents;
};

class Layer {
public:
  Layer(LayerDesc desc) {
    layer_type = desc.type;
    shape_description = desc.shape_descriptor;
    int num_weights;
    int num_weights_and_biases;
    int output_size;
    if (layer_type == CONV_LAYER) {
      // Here we translate the shape descriptor to real parameters:
      // WARNING: Changing the order of these values may result in problems in
      // other parts of code. Be careful and check all accesses of
      // shape_description
      int i = 0;
      in_height = shape_description[i++];
      in_width = shape_description[i++];
      in_depth = shape_description[i++];
      in_channels = shape_description[i++];
      out_channels = shape_description[i++];
      kernel_size = shape_description[i++];
      padding = shape_description[i++];
      stride = shape_description[i++];
      num_weights = kernel_size * kernel_size * in_channels * out_channels;
      int num_biases = out_channels;
      num_weights_and_biases = num_weights + num_biases;
      out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
      out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
      output_size = out_height * out_width * out_channels;
      input_shape = {in_height, in_width, in_channels};
      output_shape = {out_height, out_width, out_channels};
    }
    parameters =
        getCudaPointer(num_weights_and_biases, HE, num_weights, num_weights);
     
  }
  std::vector<int> shape_description;
  std::vector<int> input_shape;
  std::vector<int> output_shape;
  float *parameters;
  void forward(float *input, int batch_size) {
    if (layer_type == CONV_LAYER) {
      forward_convolution(input, batch_size, true);
    }
  }
  void backward();
  float *activations;
  std::vector<Layer *> parents;
  Layer_type layer_type;

private:
  // various parameters that may or may not be filled depending on the
  // layer_type
  int in_height;
  int in_width;
  int in_depth;
  int in_channels;
  int out_channels;
  int kernel_size;
  int padding;
  int stride;
  int out_height;
  int out_width;
  void forward_convolution(float *input, int batch_size, bool use_relu_activation);
};

#endif // !UNET_LAYER
