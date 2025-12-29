#include "addProcess.h"
#include "attention_layer_functions.h"
#include "batch_norm_test.h"
#include "concat_layer_functions.h"
#include "convolution_layer_functions.h"
#include "convolution_numpy.h"
#include "convolution_test.h"
#include "cuda_lib.h"
#include "cuda_manager.h"
#include "cuda_manager_lib.h"
#include "layer.h"
#include "max_pool_functions.h"
#include "max_pool_test.h"
#include "neural_net.h"
#include "test.h"
#include "upsample_test.h"
#include "upsampling_layer_functions.h"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// This is the main module exporting functions and enum types to the neural_net
// module. Only export functions you want to expose to the end user. Only Expose
// Enums useful for declaring or describing neural nets.
PYBIND11_MODULE(neural_net, m) {
  py::class_<NN>(m, "NeuralNet")
      .def(py::init<>())
      .def("create", &NN::create)
      .def("train", &NN::train)
      .def("infer", &NN::infer)
      .def("test", &NN::test);

  py::class_<Convolution>(m, "Convolution")
      .def(py::init<>())
      .def("convolution2D", &Convolution::convolution2D)
      .def("train", &Convolution::train);

  // This may be better to include in a separate module down the road. Howerver,
  // it is small enough where it is much more convenient to just include it
  // here right now.
  py::class_<UNetTest>(m, "UNetTest")
      .def(py::init<>())
      .def("testConvolution", &UNetTest::testConvolution)
      .def("testUpsample", &UNetTest::testUpsample)
      .def("testBatchNorm", &UNetTest::testBatchNorm)
      .def("testMaxPool", &UNetTest::testMaxPool);

  py::enum_<LossFunction>(m, "LossFunction")
      .value("DICE_LOSS", LossFunction::kDiceLoss)
      .value("DIFFERENCE_LOSS", LossFunction::kDifferenceLoss)
      .export_values();

  py::enum_<Layer_type>(m, "LayerType")
      .value("MAX_POOL_LAYER", Layer_type::MAX_POOL_LAYER)
      .value("FC_LAYER", Layer_type::FC_LAYER)
      .value("CONV_LAYER", Layer_type::CONV_LAYER)
      .value("ATTENTION_LAYER", Layer_type::ATTENTION_LAYER)
      .value("BN_LAYER", Layer_type::BN_LAYER)
      .value("UPSAMPLING_LAYER", Layer_type::UPSAMPLING_LAYER)
      .value("CONCAT_LAYER", Layer_type::CONCAT_LAYER)
      .export_values();

  py::enum_<Activation_type>(m, "ActivationType")
      .value("ReLU", Activation_type::ReLU)
      .value("SOFTMAX", Activation_type::SOFTMAX)
      .value("NONE", Activation_type::NONE)
      .value("TANH", Activation_type::TANH)
      .value("SIGMOID", Activation_type::SIGMOID)
      .value("LEAKY_ReLU", Activation_type::LEAKY_ReLU);

  py::class_<LayerDesc>(m, "LayerDesc")
      .def(py::init<const Layer_type &,
                    const std::vector<int> &,
                    const std::vector<int> &,
                    const Activation_type &,
                    double>(),
           py::arg("type"),
           py::arg("shape"),
           py::arg("parents"),
           py::arg("activation") = NONE,
           py::arg("leaky_const") = 0.01)
      .def_readwrite("type", &LayerDesc::type)
      .def_readwrite("shape", &LayerDesc::descriptor)
      .def_readwrite("parents", &LayerDesc::parents)
      .def_readwrite("activation", &LayerDesc::activation)
      .def_readwrite("leaky_const", &LayerDesc::leaky_const);
}
