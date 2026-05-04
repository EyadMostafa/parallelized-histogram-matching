#ifndef SEQUENTIAL_MATCHER_H
#define SEQUENTIAL_MATCHER_H

#include <opencv2/opencv.hpp>

class SequentialMatcher {
public:
    static void computeCDF(const cv::Mat& image, float cdf_B[256], float cdf_G[256], float cdf_R[256]);
    static void generateLUTs(const float src_cdf_B[256], const float src_cdf_G[256], const float src_cdf_R[256],
                             const float ref_cdf_B[256], const float ref_cdf_G[256], const float ref_cdf_R[256],
                             uchar lut_B[256], uchar lut_G[256], uchar lut_R[256]);
    static void applyLUT(cv::Mat& frame, const uchar lut_B[256], const uchar lut_G[256], const uchar lut_R[256]);
    static void matchHistogram(cv::Mat& frame, const float ref_cdf_B[256], const float ref_cdf_G[256], const float ref_cdf_R[256]);
};

#endif // SEQUENTIAL_MATCHER_H
