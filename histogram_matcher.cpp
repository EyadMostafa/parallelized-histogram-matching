#include "histogram_matcher.h"
#include <omp.h>

void HistogramMatcher::computeCDF(const cv::Mat& image, float cdf_B[256], float cdf_G[256], float cdf_R[256]) {
    long long hist_B[256] = {0};
    long long hist_G[256] = {0};
    long long hist_R[256] = {0};

    int total_pixels = image.rows * image.cols;
    bool continuous = image.isContinuous();
    int outer_loop = continuous ? 1 : image.rows;
    int inner_loop = continuous ? total_pixels : image.cols;

    #pragma omp parallel
    {
        long long local_hist_B[256] = {0};
        long long local_hist_G[256] = {0};
        long long local_hist_R[256] = {0};

        #pragma omp for schedule(guided)
        for (int i = 0; i < outer_loop; ++i) {
            const cv::Vec3b* row_ptr = image.ptr<cv::Vec3b>(i);
            for (int j = 0; j < inner_loop; ++j) {
                cv::Vec3b pixel = row_ptr[j];
                local_hist_B[pixel[0]]++;
                local_hist_G[pixel[1]]++;
                local_hist_R[pixel[2]]++;
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < 256; ++k) {
                hist_B[k] += local_hist_B[k];
                hist_G[k] += local_hist_G[k];
                hist_R[k] += local_hist_R[k];
            }
        }
    }

    // Convert histogram to CDF (sequential)
    long long cum_B = 0, cum_G = 0, cum_R = 0;
    for (int i = 0; i < 256; ++i) {
        cum_B += hist_B[i];
        cum_G += hist_G[i];
        cum_R += hist_R[i];
        cdf_B[i] = static_cast<float>(cum_B) / total_pixels;
        cdf_G[i] = static_cast<float>(cum_G) / total_pixels;
        cdf_R[i] = static_cast<float>(cum_R) / total_pixels;
    }
}

void HistogramMatcher::generateLUTs(const float src_cdf_B[256], const float src_cdf_G[256], const float src_cdf_R[256],
                                     const float ref_cdf_B[256], const float ref_cdf_G[256], const float ref_cdf_R[256],
                                     uchar lut_B[256], uchar lut_G[256], uchar lut_R[256]) {
    int j_B = 0, j_G = 0, j_R = 0;
    for (int i = 0; i < 256; ++i) {
        while (j_B < 255 && ref_cdf_B[j_B] < src_cdf_B[i]) j_B++;
        while (j_G < 255 && ref_cdf_G[j_G] < src_cdf_G[i]) j_G++;
        while (j_R < 255 && ref_cdf_R[j_R] < src_cdf_R[i]) j_R++;

        lut_B[i] = static_cast<uchar>(j_B);
        lut_G[i] = static_cast<uchar>(j_G);
        lut_R[i] = static_cast<uchar>(j_R);
    }
}

void HistogramMatcher::applyLUT(cv::Mat& frame, const uchar lut_B[256], const uchar lut_G[256], const uchar lut_R[256]) {
    bool continuous = frame.isContinuous();
    int outer_loop = continuous ? 1 : frame.rows;
    int inner_loop = continuous ? frame.rows * frame.cols : frame.cols;

    // Embarrassingly parallel LUT application
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < outer_loop; ++i) {
        cv::Vec3b* row_ptr = frame.ptr<cv::Vec3b>(i);
        for (int j = 0; j < inner_loop; ++j) {
            cv::Vec3b& pixel = row_ptr[j];
            pixel[0] = lut_B[pixel[0]];
            pixel[1] = lut_G[pixel[1]];
            pixel[2] = lut_R[pixel[2]];
        }
    }
}

void HistogramMatcher::matchHistogram(cv::Mat& frame, const float ref_cdf_B[256], const float ref_cdf_G[256], const float ref_cdf_R[256]) {
    float src_cdf_B[256], src_cdf_G[256], src_cdf_R[256];
    computeCDF(frame, src_cdf_B, src_cdf_G, src_cdf_R);

    uchar lut_B[256], lut_G[256], lut_R[256];
    generateLUTs(src_cdf_B, src_cdf_G, src_cdf_R, ref_cdf_B, ref_cdf_G, ref_cdf_R, lut_B, lut_G, lut_R);

    applyLUT(frame, lut_B, lut_G, lut_R);
}
