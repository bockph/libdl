//
// Created by pbo on 24.07.19.
//

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

namespace py = pybind11;


int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(libdl,m)
{
    m.doc() = "DeepLearning Lib python module";

    // Add bindings here
    m.def("foo", []() {
        return "Hello, World!";
    });

    m.def("add", &add, "A function which adds two numbers");

}