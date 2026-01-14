#pragma once

#include <opencv2/opencv.hpp>

namespace aneurysm {

class Normalizer {
public:
    Normalizer(double clahe_clip_limit = 2.0, int tile_size = 8);

    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    cv::Mat applyCLAHE(const cv::Mat& image);

    // Min-max normalization to [0, 1]
    cv::Mat normalizeMinMax(const cv::Mat& image);

    // Z-score normalization (mean=0, std=1)
    cv::Mat normalizeZScore(const cv::Mat& image);

    // Convert to grayscale if needed
    cv::Mat toGrayscale(const cv::Mat& image);

    // Resize image to target dimensions
    cv::Mat resize(const cv::Mat& image, int width, int height);

private:
    cv::Ptr<cv::CLAHE> clahe_;
    double clip_limit_;
    int tile_size_;
};

}  // namespace aneurysm
