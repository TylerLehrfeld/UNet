#ifndef UNET_NN_IMPL
#define UNET_NN_IMPL
#include "layer.h"
#include <cassert>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>
#include <stdexcept>

namespace py = pybind11;
using std::vector;

class NN {
public:
  void infer() {};
  void create(const vector<LayerDesc> &desc) {
    for (LayerDesc d : desc) {
      layers.push_back(Layer(d));
    }
    for (int i = 0; i < layers.size(); i++) {
      for (int j = 0; j < desc[i].parents.size(); j++) {
        if (desc[i].parents[j] < 0 || desc[i].parents[j] >= desc.size())
          throw std::runtime_error("invalid parent index");
        layers[i].parents.push_back(&layers[desc[i].parents[j]]);
      }
    }
    // find output layer:
    std::set<int> is_parent;
    for (LayerDesc d : desc) {
      for (int parent : d.parents) {
        is_parent.insert(parent);
      }
    }
    if (is_parent.size() != desc.size() - 1) {
      throw std::runtime_error(
          "your network does not have a single output node");
    }
    int xorAll = 0;
    int xorSet = 0;

    for (int i = 0; i < desc.size(); ++i)
      xorAll ^= i;

    for (int x : is_parent)
      xorSet ^= x;
    final_layer = xorAll ^ xorSet;
  };
  void train(py::array_t<float> images, vector<float> shape,
             py::array_t<uint8_t> masks, float learning_rate, int epochs,
             int batch_size) {
    py::buffer_info img_info = images.request();
    py::buffer_info msk_info = masks.request();

    float *img_ptr = static_cast<float *>(img_info.ptr);
    float *msk_ptr = static_cast<float *>(msk_info.ptr);
    int N = img_info.shape[0];
    int C = img_info.shape[1];
    int H = img_info.shape[2];
    int W = img_info.shape[3];
    if ((N != msk_info.shape[0]) || (H != msk_info.shape[1]) ||
        (W != msk_info.shape[2])) {
      throw std::runtime_error("Mask dimensions do not equal image dimensions");
    }
    // std::cout << "recieved images" << std::endl;
    // int n = 2;
    // int c = 1;
    // int h = 40;
    // int w = 30;
    // std::cout << img_ptr[n * C * H * W + c * H * W + h * W + w] << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
      std::cout << "beggining epoch " << epoch << "/" << epochs << std::endl;
      float total_loss = 0;
      float total_dice_score = 0;
      for (int batch_start_ind = 0; batch_start_ind < N;
           batch_start_ind += batch_size) {
        std::cout << "batch: " << batch_start_ind / batch_size << std::endl;
        int cur_batch_size = batch_size;
        if (batch_start_ind + batch_size > N) {
          cur_batch_size = N - batch_start_ind;
        }
        layers[final_layer].forward(img_ptr + batch_start_ind * C * H * W,
                                    batch_size);
        float *output_map = layers[final_layer].activations;
        float dice_score;
        float loss_val = loss(output_map, msk_ptr + batch_start_ind * H * W, H,
                              W, dice_score);
        total_dice_score += dice_score;
        total_loss += loss_val;
        backward(loss_val);
      }
      std::cout << "ephoch " << epoch << " done." << std::endl;
      std::cout << "average loss: " << total_loss / N << std::endl;
      std::cout << "average dice score: " << total_dice_score / N << std::endl;
    } // end epoch
  };

private:
  vector<Layer> layers;
  void forward(float *input, int batch_size) {
    layers[final_layer].forward(input, batch_size);
    layers[final_layer].activations;
  }
  void backward(float loss_val) {}
  float loss(float *final_map, float *y, int H, int W, float &dice_score) {
    return 0;
  }
  int final_layer;
};

#endif
