#include "roi_extractor.hpp"

namespace aneurysm {

ROIExtractor::ROIExtractor(int min_area, int max_area, double circularity_threshold)
    : min_area_(min_area), max_area_(max_area),
      circularity_threshold_(circularity_threshold) {}

std::vector<ROI> ROIExtractor::extract(const cv::Mat& image) {
    std::vector<ROI> rois;

    // Get vessel mask
    cv::Mat mask = getVesselMask(image);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Process each contour
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);

        // Filter by area
        if (area < min_area_ || area > max_area_) {
            continue;
        }

        double circularity = calculateCircularity(contour);

        // Create ROI
        ROI roi;
        roi.bbox = cv::boundingRect(contour);
        roi.circularity = circularity;
        roi.area = area;

        // Calculate center using moments
        cv::Moments m = cv::moments(contour);
        if (m.m00 > 0) {
            roi.center = cv::Point2f(
                static_cast<float>(m.m10 / m.m00),
                static_cast<float>(m.m01 / m.m00)
            );
        }

        rois.push_back(roi);
    }

    return filterByShape(rois);
}

cv::Mat ROIExtractor::getVesselMask(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Apply Gaussian blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Adaptive thresholding for vessel segmentation
    cv::Mat binary;
    cv::adaptiveThreshold(blurred, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 11, 2);

    // Morphological operations to clean up
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    return binary;
}

std::vector<ROI> ROIExtractor::filterByShape(const std::vector<ROI>& rois) {
    std::vector<ROI> filtered;

    for (const auto& roi : rois) {
        // Filter by circularity - aneurysms tend to be more circular
        if (roi.circularity >= circularity_threshold_) {
            filtered.push_back(roi);
        }
    }

    // Sort by circularity (most circular first)
    std::sort(filtered.begin(), filtered.end(),
              [](const ROI& a, const ROI& b) {
                  return a.circularity > b.circularity;
              });

    return filtered;
}

std::vector<cv::Mat> ROIExtractor::extractPatches(const cv::Mat& image,
                                                   const std::vector<ROI>& rois,
                                                   int patch_size) {
    std::vector<cv::Mat> patches;

    for (const auto& roi : rois) {
        // Calculate patch boundaries centered on ROI
        int cx = static_cast<int>(roi.center.x);
        int cy = static_cast<int>(roi.center.y);
        int half = patch_size / 2;

        int x1 = std::max(0, cx - half);
        int y1 = std::max(0, cy - half);
        int x2 = std::min(image.cols, cx + half);
        int y2 = std::min(image.rows, cy + half);

        // Extract and resize patch
        cv::Mat patch = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        cv::resize(patch, patch, cv::Size(patch_size, patch_size));
        patches.push_back(patch);
    }

    return patches;
}

double ROIExtractor::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);

    if (perimeter < 1e-6) {
        return 0.0;
    }

    // Circularity = 4 * pi * area / perimeter^2
    // Perfect circle = 1.0
    return (4.0 * CV_PI * area) / (perimeter * perimeter);
}

}  // namespace aneurysm
