#include "layer.h"
#include "neural_net.h"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(neural_net, m) {
  py::class_<NN>(m, "NeuralNet")
      .def(py::init<>())
      .def("create", &NN::create)
      .def("train", &NN::train)
      .def("infer", &NN::infer);
  py::enum_<Layer_type>(m, "LayerType")
      .value("MAX_POOL_LAYER", Layer_type::MAX_POOL_LAYER)
      .value("FC_LAYER", Layer_type::FC_LAYER)
      .value("CONV_LAYER", Layer_type::CONV_LAYER)
      .value("ATTENTION_LAYER", Layer_type::ATTENTION_LAYER)
      .value("BN_LAYER", Layer_type::BN_LAYER)
      .value("UPSAMPLING_LAYER", Layer_type::UPSAMPLING_LAYER)
      .value("CONCAT_LAYER", Layer_type::CONCAT_LAYER)
      .export_values();

  py::class_<LayerDesc>(m, "LayerDesc")
      .def(py::init<const Layer_type &, const std::vector<int> &,
                    const std::vector<int> &>(),
           py::arg("type"), py::arg("shape"), py::arg("parents"))
      .def_readwrite("type", &LayerDesc::type)
      .def_readwrite("shape", &LayerDesc::shape_descriptor)
      .def_readwrite("parents", &LayerDesc::parents);
}
