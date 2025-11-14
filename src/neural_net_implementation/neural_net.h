#ifndef UNET_NN_IMPL
#define UNET_NN_IMPL
#include "layer.h"
#include <iostream>
#include <set>
#include <stdexcept>
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
  void train() {};

private:
  vector<Layer> layers;
  void forward(float* input, int batch_size) {
    layers[final_layer].forward(input, batch_size);
    layers[final_layer].activations;
  }
  void backward();
  int final_layer;
};

#endif
