#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "image_loader.hpp"
#include "normalizer.hpp"
#include "roi_extractor.hpp"

namespace aneurysm {

struct ProcessedImage {
    cv::Mat original;        // Original image
    cv::Mat grayscale;       // Grayscale version
    cv::Mat enhanced;        // After CLAHE
    cv::Mat normalized;      // Normalized [0,1] float32
    cv::Mat resized;         // Resized to target dimensions
    std::vector<ROI> rois;   // Extracted ROIs
};

struct PreprocessorConfig {
    int target_width = 224;
    int target_height = 224;
    double clahe_clip_limit = 2.0;
    int clahe_tile_size = 8;
    bool apply_clahe = true;
    std::string normalization = "minmax";  // "minmax" or "zscore"
};

class Preprocessor {
public:
    explicit Preprocessor(const PreprocessorConfig& config = PreprocessorConfig());
    Preprocessor(int target_width, int target_height);

    // Full preprocessing pipeline
    ProcessedImage process(const std::string& image_path);
    ProcessedImage process(const cv::Mat& image);

    // Get tensor ready for ONNX inference (NCHW format, float32)
    std::vector<float> toTensor(const cv::Mat& image);

    // Convenience method: file -> tensor
    std::vector<float> preprocessToTensor(const std::string& image_path);

    // Individual operations (for flexibility)
    cv::Mat load(const std::string& path);
    cv::Mat resize(const cv::Mat& image);
    cv::Mat applyCLAHE(const cv::Mat& image);
    cv::Mat normalize(const cv::Mat& image);

    // Getters
    int getTargetWidth() const { return config_.target_width; }
    int getTargetHeight() const { return config_.target_height; }

private:
    PreprocessorConfig config_;
    ImageLoader loader_;
    Normalizer normalizer_;
    ROIExtractor roi_extractor_;
};

}  // namespace aneurysm
