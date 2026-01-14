#include "image_loader.hpp"
#include <filesystem>

namespace aneurysm {

cv::Mat ImageLoader::load(const std::string& path) {
    if (!isValidImage(path)) {
        throw std::runtime_error("Invalid image path or unsupported format: " + path);
    }

    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    return image;
}

cv::Mat ImageLoader::loadFromBuffer(const std::vector<uint8_t>& buffer) {
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to decode image from buffer");
    }
    return image;
}

bool ImageLoader::isValidImage(const std::string& path) {
    namespace fs = std::filesystem;

    if (!fs::exists(path)) {
        return false;
    }

    // Check extension
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    static const std::vector<std::string> valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };

    for (const auto& valid_ext : valid_extensions) {
        if (ext == valid_ext) {
            return true;
        }
    }

    return false;
}

std::pair<int, int> ImageLoader::getImageSize(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("Cannot read image dimensions: " + path);
    }
    return {image.cols, image.rows};
}

}  // namespace aneurysm
