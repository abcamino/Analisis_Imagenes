#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "preprocessor.hpp"

namespace py = pybind11;

// Convert cv::Mat to numpy array
py::array_t<float> mat_to_numpy_float(const cv::Mat& mat) {
    cv::Mat float_mat;
    if (mat.type() != CV_32F) {
        mat.convertTo(float_mat, CV_32F);
    } else {
        float_mat = mat;
    }

    // Create numpy array with correct shape
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (float_mat.channels() == 1) {
        shape = {float_mat.rows, float_mat.cols};
        strides = {static_cast<ssize_t>(float_mat.cols * sizeof(float)),
                   static_cast<ssize_t>(sizeof(float))};
    } else {
        shape = {float_mat.rows, float_mat.cols, float_mat.channels()};
        strides = {static_cast<ssize_t>(float_mat.cols * float_mat.channels() * sizeof(float)),
                   static_cast<ssize_t>(float_mat.channels() * sizeof(float)),
                   static_cast<ssize_t>(sizeof(float))};
    }

    auto result = py::array_t<float>(shape, strides);
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);

    // Copy data
    std::memcpy(ptr, float_mat.data, float_mat.total() * float_mat.elemSize());

    return result;
}

py::array_t<uint8_t> mat_to_numpy_uint8(const cv::Mat& mat) {
    cv::Mat uint8_mat;
    if (mat.type() != CV_8U && mat.type() != CV_8UC3) {
        mat.convertTo(uint8_mat, CV_8U, 255.0);
    } else {
        uint8_mat = mat;
    }

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (uint8_mat.channels() == 1) {
        shape = {uint8_mat.rows, uint8_mat.cols};
        strides = {static_cast<ssize_t>(uint8_mat.cols), 1};
    } else {
        shape = {uint8_mat.rows, uint8_mat.cols, uint8_mat.channels()};
        strides = {static_cast<ssize_t>(uint8_mat.cols * uint8_mat.channels()),
                   static_cast<ssize_t>(uint8_mat.channels()), 1};
    }

    auto result = py::array_t<uint8_t>(shape, strides);
    auto buf = result.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    std::memcpy(ptr, uint8_mat.data, uint8_mat.total() * uint8_mat.elemSize());

    return result;
}

// Convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t>& arr) {
    py::buffer_info buf = arr.request();

    int rows, cols, type;
    if (buf.ndim == 2) {
        rows = static_cast<int>(buf.shape[0]);
        cols = static_cast<int>(buf.shape[1]);
        type = CV_8UC1;
    } else if (buf.ndim == 3) {
        rows = static_cast<int>(buf.shape[0]);
        cols = static_cast<int>(buf.shape[1]);
        int channels = static_cast<int>(buf.shape[2]);
        type = CV_8UC(channels);
    } else {
        throw std::runtime_error("Invalid array dimensions");
    }

    cv::Mat mat(rows, cols, type, buf.ptr);
    return mat.clone();  // Clone to ensure data ownership
}

// Python module definition
PYBIND11_MODULE(aneurysm_cpp, m) {
    m.doc() = "C++ preprocessing module for aneurysm detection";

    // ROI struct
    py::class_<aneurysm::ROI>(m, "ROI")
        .def(py::init<>())
        .def_readwrite("circularity", &aneurysm::ROI::circularity)
        .def_readwrite("area", &aneurysm::ROI::area)
        .def_property("bbox",
            [](const aneurysm::ROI& r) {
                return py::make_tuple(r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height);
            },
            [](aneurysm::ROI& r, py::tuple t) {
                r.bbox = cv::Rect(t[0].cast<int>(), t[1].cast<int>(),
                                  t[2].cast<int>(), t[3].cast<int>());
            })
        .def_property("center",
            [](const aneurysm::ROI& r) {
                return py::make_tuple(r.center.x, r.center.y);
            },
            [](aneurysm::ROI& r, py::tuple t) {
                r.center = cv::Point2f(t[0].cast<float>(), t[1].cast<float>());
            });

    // PreprocessorConfig struct
    py::class_<aneurysm::PreprocessorConfig>(m, "PreprocessorConfig")
        .def(py::init<>())
        .def_readwrite("target_width", &aneurysm::PreprocessorConfig::target_width)
        .def_readwrite("target_height", &aneurysm::PreprocessorConfig::target_height)
        .def_readwrite("clahe_clip_limit", &aneurysm::PreprocessorConfig::clahe_clip_limit)
        .def_readwrite("clahe_tile_size", &aneurysm::PreprocessorConfig::clahe_tile_size)
        .def_readwrite("apply_clahe", &aneurysm::PreprocessorConfig::apply_clahe)
        .def_readwrite("normalization", &aneurysm::PreprocessorConfig::normalization);

    // ProcessedImage struct
    py::class_<aneurysm::ProcessedImage>(m, "ProcessedImage")
        .def(py::init<>())
        .def_property_readonly("original",
            [](const aneurysm::ProcessedImage& p) { return mat_to_numpy_uint8(p.original); })
        .def_property_readonly("grayscale",
            [](const aneurysm::ProcessedImage& p) { return mat_to_numpy_uint8(p.grayscale); })
        .def_property_readonly("enhanced",
            [](const aneurysm::ProcessedImage& p) { return mat_to_numpy_uint8(p.enhanced); })
        .def_property_readonly("normalized",
            [](const aneurysm::ProcessedImage& p) { return mat_to_numpy_float(p.normalized); })
        .def_property_readonly("resized",
            [](const aneurysm::ProcessedImage& p) { return mat_to_numpy_uint8(p.resized); })
        .def_readonly("rois", &aneurysm::ProcessedImage::rois);

    // Preprocessor class
    py::class_<aneurysm::Preprocessor>(m, "Preprocessor")
        .def(py::init<>())
        .def(py::init<const aneurysm::PreprocessorConfig&>(), py::arg("config"))
        .def(py::init<int, int>(), py::arg("width") = 224, py::arg("height") = 224)
        .def("process_file", py::overload_cast<const std::string&>(&aneurysm::Preprocessor::process),
             py::arg("image_path"),
             "Process image file and return ProcessedImage struct")
        .def("preprocess_to_tensor", &aneurysm::Preprocessor::preprocessToTensor,
             py::arg("image_path"),
             "Preprocess image and return flat tensor for ONNX inference")
        .def("load", &aneurysm::Preprocessor::load, py::arg("path"),
             "Load image from file path")
        .def("get_target_width", &aneurysm::Preprocessor::getTargetWidth)
        .def("get_target_height", &aneurysm::Preprocessor::getTargetHeight);

    // Convenience functions
    m.def("preprocess_image", [](const std::string& path, int width, int height) {
        aneurysm::Preprocessor p(width, height);
        auto result = p.process(path);
        return mat_to_numpy_float(result.normalized);
    }, py::arg("path"), py::arg("width") = 224, py::arg("height") = 224,
       "Quick preprocess: load, enhance, normalize, return numpy array");

    m.def("preprocess_to_tensor", [](const std::string& path, int width, int height) {
        aneurysm::Preprocessor p(width, height);
        return p.preprocessToTensor(path);
    }, py::arg("path"), py::arg("width") = 224, py::arg("height") = 224,
       "Preprocess and return flat float vector for ONNX");

    m.def("extract_rois", [](const std::string& path) {
        aneurysm::Preprocessor p;
        auto result = p.process(path);
        return result.rois;
    }, py::arg("path"),
       "Extract ROI candidates from image");

    // Version info
    m.attr("__version__") = "1.0.0";
}
