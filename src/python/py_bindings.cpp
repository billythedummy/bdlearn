#include <pybind11/pybind11.h>
#include <cstddef>
#include "bdlearn/BMat.hpp"

namespace py = pybind11;
using namespace bdlearn;

PYBIND11_MODULE(bdlearnpy, m) {
    py::class_<BMat>(m, "BMat")
        .def(py::init<size_t, size_t>());
}