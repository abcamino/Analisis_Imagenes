#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace aneurysm {

struct ROI {
    cv::Rect bbox;
    double circularity;
    double area;
    cv::Point2f center;
};

class ROIExtractor {
public:
    ROIExtractor(int min_area = 100, int max_area = 5000,
                 double circularity_threshold = 0.5);

    // Extract candidate ROIs (potential aneurysm regions)
    std::vector<ROI> extract(const cv::Mat& image);

    // Get vessel mask using adaptive thresholding
    cv::Mat getVesselMask(const cv::Mat& image);

    // Filter ROIs by shape characteristics
    std::vector<ROI> filterByShape(const std::vector<ROI>& rois);

    // Extract image patches for each ROI
    std::vector<cv::Mat> extractPatches(const cv::Mat& image,
                                         const std::vector<ROI>& rois,
                                         int patch_size = 64);

private:
    int min_area_;
    int max_area_;
    double circularity_threshold_;

    // Calculate circularity of a contour
    double calculateCircularity(const std::vector<cv::Point>& contour);
};

}  // namespace aneurysm
