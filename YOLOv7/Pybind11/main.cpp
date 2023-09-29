#include <stdio.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <opencv2/opencv.hpp>
#include "yolov7/yolov7.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(object_detection, module_handle) {
  module_handle.doc() = "Python binding for simple object detection in video with yolov7";
  py::class_<YOLOV7>(module_handle, "YOLOV7").def(py::init<int, int, int, int>())
    .def("init", &YOLOV7::init)
    .def("detect", &YOLOV7::detect)
    .def("read_rtsp", &YOLOV7::read_rtsp);
}