/*
 * Simplified C++ preprocessor module using only NumPy arrays
 * No OpenCV dependency - for easy compilation on Windows
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// Use int64_t instead of ssize_t for Windows compatibility
using ssize_t = std::ptrdiff_t;

// Simple image resize using bilinear interpolation
py::array_t<float> resize_image(py::array_t<uint8_t> input, int new_height, int new_width) {
    auto buf = input.request();

    if (buf.ndim != 2 && buf.ndim != 3) {
        throw std::runtime_error("Input must be 2D or 3D array");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    // Output array
    std::vector<ssize_t> shape = {new_height, new_width};
    if (channels > 1) shape.push_back(channels);

    auto result = py::array_t<float>(shape);
    auto res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    float y_ratio = static_cast<float>(height - 1) / (new_height - 1);
    float x_ratio = static_cast<float>(width - 1) / (new_width - 1);

    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            float y = i * y_ratio;
            float x = j * x_ratio;

            int y0 = static_cast<int>(y);
            int x0 = static_cast<int>(x);
            int y1 = std::min(y0 + 1, height - 1);
            int x1 = std::min(x0 + 1, width - 1);

            float dy = y - y0;
            float dx = x - x0;

            for (int c = 0; c < channels; c++) {
                float v00, v01, v10, v11;

                if (channels == 1) {
                    v00 = ptr[y0 * width + x0];
                    v01 = ptr[y0 * width + x1];
                    v10 = ptr[y1 * width + x0];
                    v11 = ptr[y1 * width + x1];
                } else {
                    v00 = ptr[(y0 * width + x0) * channels + c];
                    v01 = ptr[(y0 * width + x1) * channels + c];
                    v10 = ptr[(y1 * width + x0) * channels + c];
                    v11 = ptr[(y1 * width + x1) * channels + c];
                }

                float value = v00 * (1 - dx) * (1 - dy) +
                              v01 * dx * (1 - dy) +
                              v10 * (1 - dx) * dy +
                              v11 * dx * dy;

                if (channels == 1) {
                    res_ptr[i * new_width + j] = value / 255.0f;
                } else {
                    res_ptr[(i * new_width + j) * channels + c] = value / 255.0f;
                }
            }
        }
    }

    return result;
}

// Convert RGB to grayscale
py::array_t<float> to_grayscale(py::array_t<uint8_t> input) {
    auto buf = input.request();

    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input must be 3D array with 3 channels");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    auto result = py::array_t<float>({height, width});
    auto res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            // Standard luminosity formula (BGR order from OpenCV)
            float gray = 0.114f * ptr[idx] + 0.587f * ptr[idx + 1] + 0.299f * ptr[idx + 2];
            res_ptr[i * width + j] = gray / 255.0f;
        }
    }

    return result;
}

// Min-max normalization
py::array_t<float> normalize_minmax(py::array_t<float> input) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    float min_val = *std::min_element(ptr, ptr + size);
    float max_val = *std::max_element(ptr, ptr + size);

    auto result = py::array_t<float>(buf.shape);
    auto res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    for (size_t i = 0; i < size; i++) {
        res_ptr[i] = (ptr[i] - min_val) / range;
    }

    return result;
}

// Simple CLAHE-like contrast enhancement
py::array_t<float> enhance_contrast(py::array_t<float> input, float clip_limit = 2.0f) {
    auto buf = input.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D grayscale image");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    float* ptr = static_cast<float*>(buf.ptr);

    auto result = py::array_t<float>({height, width});
    auto res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    // Simple local contrast enhancement (simplified CLAHE)
    int tile_size = 8;
    int tile_h = (height + tile_size - 1) / tile_size;
    int tile_w = (width + tile_size - 1) / tile_size;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Define local region
            int y0 = std::max(0, i - tile_size / 2);
            int y1 = std::min(height, i + tile_size / 2);
            int x0 = std::max(0, j - tile_size / 2);
            int x1 = std::min(width, j + tile_size / 2);

            // Calculate local statistics
            float sum = 0, sum_sq = 0;
            int count = 0;

            for (int yi = y0; yi < y1; yi++) {
                for (int xi = x0; xi < x1; xi++) {
                    float val = ptr[yi * width + xi];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }

            float mean = sum / count;
            float variance = (sum_sq / count) - (mean * mean);
            float std_dev = std::sqrt(std::max(variance, 1e-6f));

            // Apply contrast enhancement with clipping
            float pixel = ptr[i * width + j];
            float enhanced = (pixel - mean) / (std_dev * clip_limit) * 0.5f + 0.5f;
            res_ptr[i * width + j] = std::max(0.0f, std::min(1.0f, enhanced));
        }
    }

    return result;
}

// Complete preprocessing pipeline
py::dict preprocess_image(py::array_t<uint8_t> input, int target_height = 224, int target_width = 224) {
    py::dict result;

    auto buf = input.request();
    bool is_color = (buf.ndim == 3 && buf.shape[2] == 3);

    // Convert to grayscale if needed
    py::array_t<float> gray;
    if (is_color) {
        gray = to_grayscale(input);
    } else {
        // Already grayscale, just convert to float
        auto gbuf = input.request();
        int h = gbuf.shape[0];
        int w = gbuf.shape[1];
        uint8_t* ptr = static_cast<uint8_t*>(gbuf.ptr);

        gray = py::array_t<float>({h, w});
        auto res_buf = gray.request();
        float* res_ptr = static_cast<float*>(res_buf.ptr);

        for (int i = 0; i < h * w; i++) {
            res_ptr[i] = ptr[i] / 255.0f;
        }
    }

    // Enhance contrast
    py::array_t<float> enhanced = enhance_contrast(gray, 2.0f);

    // Resize
    // First convert back to uint8 for resize
    auto enh_buf = enhanced.request();
    int h = enh_buf.shape[0];
    int w = enh_buf.shape[1];
    float* enh_ptr = static_cast<float*>(enh_buf.ptr);

    py::array_t<uint8_t> enh_uint8({h, w});
    auto uint8_buf = enh_uint8.request();
    uint8_t* uint8_ptr = static_cast<uint8_t*>(uint8_buf.ptr);

    for (int i = 0; i < h * w; i++) {
        uint8_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, enh_ptr[i] * 255.0f)));
    }

    py::array_t<float> resized = resize_image(enh_uint8, target_height, target_width);

    // Normalize
    py::array_t<float> normalized = normalize_minmax(resized);

    result["grayscale"] = gray;
    result["enhanced"] = enhanced;
    result["resized"] = resized;
    result["normalized"] = normalized;

    return result;
}

PYBIND11_MODULE(aneurysm_cpp, m) {
    m.doc() = "Simplified C++ preprocessing module (no OpenCV dependency)";

    m.def("resize_image", &resize_image,
          py::arg("input"), py::arg("height"), py::arg("width"),
          "Resize image using bilinear interpolation");

    m.def("to_grayscale", &to_grayscale,
          py::arg("input"),
          "Convert RGB image to grayscale");

    m.def("normalize_minmax", &normalize_minmax,
          py::arg("input"),
          "Normalize array to [0, 1] range");

    m.def("enhance_contrast", &enhance_contrast,
          py::arg("input"), py::arg("clip_limit") = 2.0f,
          "Apply local contrast enhancement (simplified CLAHE)");

    m.def("preprocess_image", &preprocess_image,
          py::arg("input"), py::arg("target_height") = 224, py::arg("target_width") = 224,
          "Complete preprocessing pipeline: grayscale -> enhance -> resize -> normalize");

    m.attr("__version__") = "1.0.0";
}
