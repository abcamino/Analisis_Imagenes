#include "normalizer.hpp"

namespace aneurysm {

Normalizer::Normalizer(double clahe_clip_limit, int tile_size)
    : clip_limit_(clahe_clip_limit), tile_size_(tile_size) {
    clahe_ = cv::createCLAHE(clip_limit_, cv::Size(tile_size_, tile_size_));
}

cv::Mat Normalizer::applyCLAHE(const cv::Mat& image) {
    cv::Mat gray = toGrayscale(image);
    cv::Mat result;
    clahe_->apply(gray, result);
    return result;
}

cv::Mat Normalizer::normalizeMinMax(const cv::Mat& image) {
    cv::Mat result;
    cv::Mat float_image;

    // Convert to float
    image.convertTo(float_image, CV_32F);

    // Find min and max
    double min_val, max_val;
    cv::minMaxLoc(float_image, &min_val, &max_val);

    // Normalize to [0, 1]
    if (max_val - min_val > 1e-6) {
        result = (float_image - min_val) / (max_val - min_val);
    } else {
        result = cv::Mat::zeros(float_image.size(), CV_32F);
    }

    return result;
}

cv::Mat Normalizer::normalizeZScore(const cv::Mat& image) {
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(float_image, mean, stddev);

    cv::Mat result;
    if (stddev[0] > 1e-6) {
        result = (float_image - mean[0]) / stddev[0];
    } else {
        result = float_image - mean[0];
    }

    return result;
}

cv::Mat Normalizer::toGrayscale(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image.clone();
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat Normalizer::resize(const cv::Mat& image, int width, int height) {
    cv::Mat result;
    cv::resize(image, result, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    return result;
}

}  // namespace aneurysm
