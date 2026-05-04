#ifndef HISTOGRAM_MATCHER_H
#define HISTOGRAM_MATCHER_H

#include <opencv2/opencv.hpp>

class HistogramMatcher {
public:
    // Computes CDF for a given image (OpenMP parallelized)
    static void computeCDF(const cv::Mat& image, float cdf_B[256], float cdf_G[256], float cdf_R[256]);
    
    // Generates Lookup Tables mapping source CDF to reference CDF (Sequential, O(N))
    static void generateLUTs(const float src_cdf_B[256], const float src_cdf_G[256], const float src_cdf_R[256],
                             const float ref_cdf_B[256], const float ref_cdf_G[256], const float ref_cdf_R[256],
                             uchar lut_B[256], uchar lut_G[256], uchar lut_R[256]);
                             
    // Applies LUT to an image frame using OpenMP
    static void applyLUT(cv::Mat& frame, const uchar lut_B[256], const uchar lut_G[256], const uchar lut_R[256]);

    // Matches the histogram of 'frame' to the precomputed 'ref_cdf'
    static void matchHistogram(cv::Mat& frame, const float ref_cdf_B[256], const float ref_cdf_G[256], const float ref_cdf_R[256]);
};

#endif // HISTOGRAM_MATCHER_H
