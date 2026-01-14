#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

namespace aneurysm {

class ImageLoader {
public:
    ImageLoader() = default;

    // Load image from file path
    cv::Mat load(const std::string& path);

    // Load image from memory buffer
    cv::Mat loadFromBuffer(const std::vector<uint8_t>& buffer);

    // Check if file exists and is valid image
    bool isValidImage(const std::string& path);

    // Get image dimensions without loading full image
    std::pair<int, int> getImageSize(const std::string& path);
};

}  // namespace aneurysm
