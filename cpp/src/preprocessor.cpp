#include "preprocessor.hpp"

namespace aneurysm {

Preprocessor::Preprocessor(const PreprocessorConfig& config)
    : config_(config),
      normalizer_(config.clahe_clip_limit, config.clahe_tile_size) {}

Preprocessor::Preprocessor(int target_width, int target_height)
    : config_{target_width, target_height},
      normalizer_(config_.clahe_clip_limit, config_.clahe_tile_size) {}

ProcessedImage Preprocessor::process(const std::string& image_path) {
    cv::Mat image = loader_.load(image_path);
    return process(image);
}

ProcessedImage Preprocessor::process(const cv::Mat& image) {
    ProcessedImage result;

    // Store original
    result.original = image.clone();

    // Convert to grayscale
    result.grayscale = normalizer_.toGrayscale(image);

    // Apply CLAHE if enabled
    if (config_.apply_clahe) {
        result.enhanced = normalizer_.applyCLAHE(result.grayscale);
    } else {
        result.enhanced = result.grayscale.clone();
    }

    // Resize to target dimensions
    result.resized = normalizer_.resize(result.enhanced,
                                         config_.target_width,
                                         config_.target_height);

    // Normalize
    if (config_.normalization == "zscore") {
        result.normalized = normalizer_.normalizeZScore(result.resized);
    } else {
        result.normalized = normalizer_.normalizeMinMax(result.resized);
    }

    // Extract ROIs from enhanced image
    result.rois = roi_extractor_.extract(result.enhanced);

    return result;
}

std::vector<float> Preprocessor::toTensor(const cv::Mat& image) {
    cv::Mat processed;

    // Ensure float32 and normalized
    if (image.type() != CV_32F) {
        image.convertTo(processed, CV_32F);
        processed = normalizer_.normalizeMinMax(processed);
    } else {
        processed = image.clone();
    }

    // Resize if needed
    if (processed.rows != config_.target_height ||
        processed.cols != config_.target_width) {
        cv::resize(processed, processed,
                   cv::Size(config_.target_width, config_.target_height));
    }

    // Convert to NCHW format for ONNX (1, 1, H, W) for grayscale
    // or (1, 3, H, W) for RGB
    std::vector<float> tensor;

    if (processed.channels() == 1) {
        // Grayscale: (1, 1, H, W)
        tensor.resize(config_.target_height * config_.target_width);

        for (int y = 0; y < processed.rows; ++y) {
            for (int x = 0; x < processed.cols; ++x) {
                tensor[y * processed.cols + x] = processed.at<float>(y, x);
            }
        }
    } else {
        // RGB: (1, 3, H, W) - channels first
        int channel_size = config_.target_height * config_.target_width;
        tensor.resize(3 * channel_size);

        std::vector<cv::Mat> channels;
        cv::split(processed, channels);

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < processed.rows; ++y) {
                for (int x = 0; x < processed.cols; ++x) {
                    tensor[c * channel_size + y * processed.cols + x] =
                        channels[c].at<float>(y, x);
                }
            }
        }
    }

    return tensor;
}

std::vector<float> Preprocessor::preprocessToTensor(const std::string& image_path) {
    ProcessedImage processed = process(image_path);
    return toTensor(processed.normalized);
}

cv::Mat Preprocessor::load(const std::string& path) {
    return loader_.load(path);
}

cv::Mat Preprocessor::resize(const cv::Mat& image) {
    return normalizer_.resize(image, config_.target_width, config_.target_height);
}

cv::Mat Preprocessor::applyCLAHE(const cv::Mat& image) {
    return normalizer_.applyCLAHE(image);
}

cv::Mat Preprocessor::normalize(const cv::Mat& image) {
    if (config_.normalization == "zscore") {
        return normalizer_.normalizeZScore(image);
    }
    return normalizer_.normalizeMinMax(image);
}

}  // namespace aneurysm
