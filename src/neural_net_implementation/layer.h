#ifndef UNET_LAYER
#define UNET_LAYER

#include "cuda_lib.h"
#include <cassert>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int shape_size(const std::vector<int> &shape) {
  int s = 1;
  for(int d : shape) {
    s *= d;
  }
  return s;
}

enum Layer_type {
  CONV_LAYER,
  BN_LAYER,
  MAX_POOL_LAYER,
  ATTENTION_LAYER,
  FC_LAYER,
  CONCAT_LAYER,
  UPSAMPLING_LAYER,
};

enum Activation_type { ReLU, SOFTMAX, NONE, TANH, SIGMOID, LEAKY_ReLU };
struct LayerDesc {
  Layer_type type;
  std::vector<int> descriptor;
  std::vector<int> parents;
  Activation_type activation;
  double leaky_const;
  LayerDesc(const Layer_type &t,
            const std::vector<int> &d,
            const std::vector<int> &p,
            const Activation_type &a,
            double lc)
      : type(t), descriptor(d), parents(p), activation(a), leaky_const(lc) {}
};

class Layer {
public:
  Layer(const LayerDesc &desc, std::vector<Layer> &parents_vec) {
    layer_type = desc.type;
    // std::cout << "creating layer. Layer type: " << layer_type << std::endl;
    description = desc.descriptor;
    // make sure each layer has access to its parents for initialization
    for(int parent : desc.parents) {
      parents.push_back(&parents_vec[parent]);
    }
    if(layer_type == CONV_LAYER) {
      init_conv_layer(desc);
    } else if(layer_type == BN_LAYER) {
      // TODO: NOT DOING FOR THIS PROJECT
    } else if(layer_type == MAX_POOL_LAYER) {
      init_max_pool(desc);
    } else if(layer_type == ATTENTION_LAYER) {
      init_attention_layer(desc);
    } else if(layer_type == FC_LAYER) {
      // TODO: NOT DOINT FOR THIS PROJECT
    } else if(layer_type == CONCAT_LAYER) {
      init_concat_layer(desc);
    } else if(layer_type == UPSAMPLING_LAYER) {
      init_upsampling_layer(desc);
    }
  }
  std::vector<int> description;
  std::vector<int> input_shape;
  std::vector<int> output_shape;
  void forward(float *input, int batch_size, bool inference) {
    // done marks whether the forward pass has been done already. This is
    // important because some layers' activations can be called for twice by
    // different children
    if(done) {
      return;
    }
    if(parents.size() != 0) {
      for(Layer *parent : parents) {
        parent->forward(input, batch_size, inference);
      }
    }
    if(layer_type == CONV_LAYER) {
      if(parents.size() == 0) {
        forward_convolution(input, batch_size, true, true, inference);
      } else {
        float *prev_layer_activations = parents[0]->activations;
        forward_convolution(
            prev_layer_activations, batch_size, true, true, inference);
      }
    } else if(layer_type == CONCAT_LAYER) {
      float *activations_1 = parents[0]->activations;
      float *activations_2 = parents[1]->activations;
      forward_concat(activations_1, activations_2, batch_size);
    } else if(layer_type == MAX_POOL_LAYER) {
      if(parents.size() == 0) {
        forward_max_pool(input, batch_size);
      } else {
        float *prev_layer_activations = parents[0]->activations;
        forward_max_pool(prev_layer_activations, batch_size);
      }
    } else if(layer_type == UPSAMPLING_LAYER) {
      if(parents.size() == 0) {
        forward_upsample(input, batch_size, true, true, inference);
      } else {
        float *prev_activations = parents[0]->activations;
        forward_upsample(prev_activations, batch_size, true, true, inference);
      }
    } else if(layer_type == ATTENTION_LAYER) {
      float *x_inputs = parents[0]->activations;
      float *g_inputs = parents[1]->activations;
      forward_attention(x_inputs, g_inputs, batch_size);
    }

    done = true;
  }
  void backward(int batch_size, float *original_inputs) {

    if(children_seen != num_children || done) {
      return;
    }
    done = true;

    init_gradient_arrays(batch_size);
    float *inputs;
    if(parents.size() == 0) {
      inputs = original_inputs;
    } else {
      inputs = parents[0]->activations;
    }
    if(layer_type == CONV_LAYER) {
      backward_convolution(dLdY_copy, inputs, batch_size);
    } else if(layer_type == CONCAT_LAYER) {
      float *inputs_2 = parents[1]->activations;
      backward_concat(dLdY_copy, inputs, inputs_2, batch_size);
    } else if(layer_type == MAX_POOL_LAYER) {
      backward_max_pool(dLdY_copy, inputs, batch_size);
    } else if(layer_type == UPSAMPLING_LAYER) {
      backward_upsample(dLdY_copy, inputs, batch_size);
    } else if(layer_type == ATTENTION_LAYER) {
      float *inputs_g = parents[1]->activations;
      backward_attention(dLdY_copy, inputs, inputs_g, batch_size);
      int input_x_size = batch_size * shape_size(parents[0]->output_shape);
    }

    int size_prev = 0;
    for(int p = 0; p < parents.size(); p++) {

      int size_dLdY = batch_size * shape_size(parents[p]->output_shape);
      if(parents[p]->children_seen == 0) {

        parents[p]->dLdY_copy = getCudaPointer(size_dLdY);

        cuda_lib_copy_device(
            dLdX + size_prev, size_dLdY, parents[p]->dLdY_copy);
      } else {
        cuda_add_nums(dLdX + size_prev, parents[p]->dLdY_copy, size_dLdY);
      }
      size_prev += batch_size * shape_size(parents[p]->output_shape);
      parents[p]->children_seen++;
    }
    free_train_arrs();
    // std::cout << "backwards for layer type: " << layer_type << std::endl;
    if(parents.size() > 0) {
      for(int p = 0; p < parents.size(); p++) {
        parents[p]->backward(batch_size, original_inputs);
      }
    }
  }
  void step(float learning_rate, int t) {
    if(layer_type == CONV_LAYER) {
      cuda_update_weights(parameters,
                          dLdW,
                          adam_weights,
                          learning_rate,
                          num_weights_and_biases,
                          t);
      cuda_update_weights(BN_parameters,
                          dLdW + num_weights_and_biases,
                          adam_weights + num_weights_and_biases * 2,
                          learning_rate,
                          num_BN_params,
                          t);
    } else if(layer_type == UPSAMPLING_LAYER) {
      cuda_update_weights(parameters,
                          dLdW,
                          adam_weights,
                          learning_rate,
                          num_weights_and_biases,
                          t);
      cuda_update_weights(BN_parameters,
                          dLdW + num_weights_and_biases,
                          adam_weights + num_weights_and_biases * 2,
                          learning_rate,
                          num_BN_params,
                          t);
    } else if(layer_type == ATTENTION_LAYER) {
      cuda_update_weights(parameters,
                          dLdW,
                          adam_weights,
                          learning_rate,
                          num_weights_and_biases,
                          t);
    }
  }
  void zero_grad(int batch_size) {

    int size_prev = batch_size * shape_size(input_shape);

    int size_out = batch_size * shape_size(output_shape);
    if(layer_type == CONV_LAYER) {

      cudaSetZero(dLdX, size_prev);
      cudaSetZero(grad_activations_int_1, size_out);
      cudaSetZero(grad_activations_int_2, size_out);
      cudaSetZero(dLdW, num_weights_and_biases + num_BN_params);
    } else if(layer_type == CONCAT_LAYER) {
      cudaSetZero(dLdX, size_prev);
    } else if(layer_type == MAX_POOL_LAYER) {
      cudaSetZero(dLdX, size_prev);
    } else if(layer_type == UPSAMPLING_LAYER) {
      cudaSetZero(dLdX, size_prev);
      cudaSetZero(grad_activations_int_1, size_out);
      cudaSetZero(grad_activations_int_2, size_out);
      cudaSetZero(dLdW, num_weights_and_biases + num_BN_params);
    } else if(layer_type == ATTENTION_LAYER) {
      cudaSetZero(dLdX, size_prev);
      cudaSetZero(grad_activations_int_1, size_out);
      cudaSetZero(grad_activations_int_2, size_out);
      cudaSetZero(dLdW, num_weights_and_biases);
    }
  }
  void free_train_arrs() {
    cudaLibFree(dLdX);
    cudaLibFree(dLdY_copy);
    if(layer_type == CONV_LAYER) {
      // cudaLibFree(dLdX);
      cudaLibFree(grad_activations_int_1);
      cudaLibFree(grad_activations_int_2);
      // cudaLibFree(dLdW);
    } else if(layer_type == CONCAT_LAYER) {
      // cudaLibFree(dLdX);
    } else if(layer_type == MAX_POOL_LAYER) {
      // cudaLibFree(dLdX);
    } else if(layer_type == UPSAMPLING_LAYER) {
      // cudaLibFree(dLdX);
      cudaLibFree(grad_activations_int_1);
      cudaLibFree(grad_activations_int_2);
      // cudaLibFree(dLdW);
    } else if(layer_type == ATTENTION_LAYER) {
      // cudaLibFree(dLdX);
      cudaLibFree(grad_activations_int_1);
      cudaLibFree(grad_activations_int_2);
      // cudaLibFree(dLdW);
    }
  }

  void init_activation_arrays_and_parameters(int batch_size) {
    int size_prev = batch_size * shape_size(input_shape);
    if(layer_type == CONV_LAYER) {
      int size_out = batch_size * shape_size(output_shape);
      adam_weights =
          getCudaPointer((num_weights_and_biases + num_BN_params) * 2);
      activations_int_1 = getCudaPointer(batch_size * shape_size(output_shape));
      activations_int_2 = getCudaPointer(batch_size * shape_size(output_shape));
      activations = getCudaPointer(batch_size * shape_size(output_shape));
      dLdW = getCudaPointer(num_weights_and_biases + num_BN_params);
    } else if(layer_type == CONCAT_LAYER) {
      activations = getCudaPointer(batch_size * shape_size(output_shape));
    } else if(layer_type == MAX_POOL_LAYER) {
      activations = getCudaPointer(batch_size * shape_size(output_shape));
    } else if(layer_type == UPSAMPLING_LAYER) {
      int size_out = batch_size * shape_size(output_shape);
      adam_weights =
          getCudaPointer((num_weights_and_biases + num_BN_params) * 2);
      activations_int_1 = getCudaPointer(size_out);
      activations_int_2 = getCudaPointer(size_out);
      activations = getCudaPointer(size_out);
      dLdW = getCudaPointer(num_weights_and_biases + num_BN_params);
    } else if(layer_type == ATTENTION_LAYER) {
      int size_prev =
          batch_size * (input_shape[0] * input_shape[1] * input_shape[2] +
                        input_shape[3] * input_shape[4] * input_shape[5]);

      adam_weights = getCudaPointer(num_weights_and_biases * 2);
      int intermediate_size =
          input_shape[0] / 2 * input_shape[1] / 2 * input_shape[2] / 2;

      activations_int_1 = getCudaPointer(batch_size * intermediate_size);

      activations_int_2 =
          getCudaPointer(batch_size * input_shape[1] / 2 * input_shape[2] / 2);

      activations = getCudaPointer(batch_size * shape_size(output_shape));
      dLdW = getCudaPointer(num_weights_and_biases);
    }
  }

  void init_gradient_arrays(int batch_size) {
    int size_prev = batch_size * shape_size(input_shape);
    if(layer_type == CONV_LAYER) {
      int size_out = batch_size * shape_size(output_shape);
      dLdX = getCudaPointer(size_prev);
      grad_activations_int_1 = getCudaPointer(size_out);
      grad_activations_int_2 = getCudaPointer(size_out);
      // dLdW = getCudaPointer(num_weights_and_biases + num_BN_params);
    } else if(layer_type == CONCAT_LAYER) {
      dLdX = getCudaPointer(size_prev);
    } else if(layer_type == MAX_POOL_LAYER) {
      dLdX = getCudaPointer(size_prev);
    } else if(layer_type == UPSAMPLING_LAYER) {
      int size_out = batch_size * shape_size(output_shape);
      dLdX = getCudaPointer(size_prev);
      // dLdW = getCudaPointer(num_weights_and_biases + num_BN_params);
      grad_activations_int_1 = getCudaPointer(size_out);
      grad_activations_int_2 = getCudaPointer(size_out);
    } else if(layer_type == ATTENTION_LAYER) {
      int size_prev =
          batch_size * (input_shape[0] * input_shape[1] * input_shape[2] +
                        input_shape[3] * input_shape[4] * input_shape[5]);
      dLdX = getCudaPointer(size_prev);
      // dLdW = getCudaPointer(num_weights_and_biases);

      int intermediate_size =
          input_shape[0] / 2 * input_shape[1] / 2 * input_shape[2] / 2;

      grad_activations_int_1 = getCudaPointer(batch_size * intermediate_size);

      grad_activations_int_2 =
          getCudaPointer(batch_size * input_shape[1] / 2 * input_shape[2] / 2);
    }
  }
  float *dLdX;
  float *dLdY_copy;
  int num_children;
  int children_seen;
  float *dLdW;
  float *parameters;
  float *BN_parameters;
  float *activations;
  std::vector<Layer *> parents;
  Layer_type layer_type;
  Activation_type activation_type;

  bool done;

private:
  // various parameters that may or may not be filled depending on the
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
  int num_weights_and_biases = 0;
  int num_BN_params = 0;
  bool batch_norm;
  float *BN_stats;
  float *BN_batch_stats;
  float *activations_int_1;
  float *activations_int_2;
  float *grad_activations_int_1;
  float *grad_activations_int_2;
  float *adam_weights;
  void forward_convolution(float *input,
                           int batch_size,
                           bool use_relu_activation,
                           bool use_batch_norm,
                           bool inference);

  void
  backward_convolution(float *grad_activations, float *input, int batch_size);
  void backward_concat(float *grad_activations,
                       float *input_1,
                       float *input_2,
                       int batch_size);

  void backward_max_pool(float *dLdY, float *inputs, int batch_size);

  void forward_upsample(float *input,
                        int batch_size,
                        bool use_relu_activation,
                        bool use_batch_norm,
                        bool inference);

  void backward_upsample(float *grad_activations, float *input, int batch_size);

  void forward_attention(float *input_x, float *input_g, int batch_size);
  void backward_attention(float *grad_activations,
                          float *input_x,
                          float *input_g,
                          int batch_size);
  void forward_max_pool(float *input, int batch_size);

  void
  forward_concat(float *activations_1, float *activations_2, int batch_size);
  void init_conv_layer(const LayerDesc &desc) {
    if(parents.size() > 1) {
      throw std::runtime_error(
          "Convolution layers cannot have more than one parent");
    }
    int num_weights;

    int output_size;
    // Here we translate the shape descriptor to real parameters:
    // WARNING: Changing the order of these values may result in problems in
    // other parts of code. Be careful and check all accesses of
    // shape_description
    int i = 0;
    int num_params = description.size();
    in_height = description[i++];
    in_width = description[i++];
    in_channels = description[i++];
    out_channels = description[i++];
    kernel_size = description[i++];
    padding = description[i++];
    stride = description[i++];
    batch_norm = true;
    activation_type = desc.activation;
    num_weights = kernel_size * kernel_size * in_channels * out_channels;
    int num_biases = out_channels;
    num_weights_and_biases = num_weights + num_biases;
    out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    output_size = out_height * out_width * out_channels;
    input_shape = {in_channels, in_height, in_width};
    output_shape = {out_channels, out_height, out_width};
    parameters =
        getCudaPointer(num_weights_and_biases, HE, num_weights, num_weights);
    BN_parameters = getCudaPointer(out_channels * 2, BN);
    num_BN_params = out_channels * 2;
    BN_stats = getCudaPointer(out_channels * 2, BN);
    BN_batch_stats = getCudaPointer(out_channels * 2);
    std::cout << "add params: " << num_BN_params + num_weights_and_biases
              << std::endl;
  }

  void init_max_pool(const LayerDesc &desc) {
    if(parents.size() > 1) {
      std::stringstream ss;
      ss << "max pool layer must have zero or one parent. Recieved "
         << parents.size() << " parents.";
      throw std::runtime_error(ss.str());
    }
    int i = 0;
    in_channels = description[i++];
    in_height = description[i++];
    in_width = description[i++];
    kernel_size = description[i++];
    stride = description[i++];

    out_height = in_height / stride;

    out_width = in_width / stride;
    input_shape = {in_channels, in_height, in_width};
    output_shape = {
        input_shape[0], input_shape[1] / stride, input_shape[2] / stride};
    if(input_shape[1] % stride != 0) {
      output_shape[1]++;
    }
    if(input_shape[2] % stride != 0) {
      output_shape[2]++;
    }
  }

  // concat along channel dimension
  void init_concat_layer(const LayerDesc &desc) {
    if(parents.size() != 2) {
      std::stringstream ss;
      ss << "Concat layer must have two parents. Current parent count is "
         << parents.size();
      throw std::runtime_error(ss.str());
    }
    int i = 0;
    if(parents[i++]->output_shape.size() != 3 ||
       parents[i++]->output_shape.size() != 3) {
      std::stringstream ss;
      ss << "Parent " << i
         << " does not have 3 shape parameters C, H, and W. Current shape "
            "size "
            "is "
         << parents[i - 1]->output_shape.size();
      throw std::runtime_error(ss.str());
    }
    if(parents[0]->output_shape[1] != parents[1]->output_shape[1] ||
       parents[0]->output_shape[2] != parents[1]->output_shape[2]) {
      std::stringstream ss;
      ss << "Concat layer must have equal heights and widths. Height and "
            "width "
            "of parent 1: "
         << parents[0]->output_shape[1] << ", " << parents[0]->output_shape[2]
         << " Height and width of parent 2: " << parents[1]->output_shape[1]
         << ", " << parents[1]->output_shape[2];
      throw std::runtime_error(ss.str());
    }
    // C_1, C_2, H, W
    input_shape = {parents[0]->output_shape[0],
                   parents[1]->output_shape[0],
                   parents[0]->output_shape[1],
                   parents[0]->output_shape[2]};

    output_shape = parents[0]->output_shape;
    output_shape[0] += parents[1]->output_shape[0];
  }

  void init_attention_layer(const LayerDesc &desc) {
    if(parents.size() != 2) {
      std::stringstream ss;
      ss << "Attention layers need two parents: " << parents.size()
         << " parents givent";
      throw std::runtime_error(ss.str());
    }
    int g_in_channels = parents[1]->out_channels;
    int x_in_channels = parents[0]->out_channels;
    int intermediate_channels = x_in_channels / 2;
    int g_H = parents[1]->out_height;
    int g_W = parents[1]->out_width;
    int x_H = parents[0]->out_height;
    int x_W = parents[0]->out_width;
    if(g_H != x_H / 2 || g_W != x_W / 2) {
      std::stringstream ss;
      ss << "attention not implemented for g that is not half the size of x. "
            "g "
            "dimensions: "
         << g_H << ", " << g_W << std::endl
         << "x dimensions: " << x_H << ", " << x_W;
      throw std::runtime_error(ss.str());
    }
    out_height = x_H;
    in_height = x_H;
    out_width = x_W;
    in_width = x_W;
    in_channels = parents[0]->output_shape[0];
    out_channels = in_channels;
    output_shape = parents[0]->output_shape;
    input_shape = {parents[0]->output_shape[0],
                   parents[0]->output_shape[1],
                   parents[0]->output_shape[2],
                   parents[1]->output_shape[0],
                   parents[1]->output_shape[1],
                   parents[1]->output_shape[2]};

    // organized like so parameters[in_channel][out_channel]
    // in channel is organized x, then g.
    // after this, we have the psi weights. Then the intermediate, and psi
    // biases.
    int num_weights = g_in_channels * intermediate_channels +
                      x_in_channels * intermediate_channels +
                      intermediate_channels;
    int num_biases = intermediate_channels + 1;
    num_weights_and_biases = num_biases + num_weights;
    // TODO determine if this is a good enough initialization
    parameters =
        getCudaPointer(num_weights_and_biases, HE, num_weights, num_weights);
    std::cout << "add params: " << num_weights_and_biases << std::endl;
  }

  void init_upsampling_layer(const LayerDesc &desc) {
    if(parents.size() > 1) {
      std::stringstream ss;
      ss << "upsampling layers need 0 or 1 parents: " << parents.size()
         << " parents givent";
      throw std::runtime_error(ss.str());
    }
    in_height = description[0];
    in_width = description[1];
    in_channels = description[2];
    out_channels = description[3];
    kernel_size = description[4];
    input_shape = {in_channels, in_height, in_width};
    int scale = kernel_size;
    out_height = scale * in_height;
    out_width = scale * in_width;
    output_shape = {out_channels, out_height, out_width};
    if(input_shape != parents[0]->output_shape) {
      std::stringstream ss;
      ss << "input shape to upsampling layer does not equal output shape of "
            "parent: "
         << std::endl
         << "upsampling input: ";
      for(int p : input_shape) {
        ss << p << ", ";
      }
      ss << std::endl << "parent output: ";
      for(int p : parents[0]->output_shape) {
        ss << p << ", ";
      }
      throw std::runtime_error(ss.str());
    }
    int num_weights = in_channels * out_channels * scale * scale;
    int num_biases = out_channels;
    num_weights_and_biases = num_weights + num_biases;
    parameters =
        getCudaPointer(num_biases + num_weights, HE, num_weights, num_weights);
    BN_parameters = getCudaPointer(out_channels * 2, BN);
    num_BN_params = out_channels * 2;
    BN_stats = getCudaPointer(out_channels * 2, BN);
    BN_batch_stats = getCudaPointer(out_channels * 2);
    std::cout << "add params: " << num_weights_and_biases + num_BN_params
              << std::endl;
  }
};

#endif // !UNET_LAYER
