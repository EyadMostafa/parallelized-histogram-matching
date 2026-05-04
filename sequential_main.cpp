#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "sequential_matcher.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_video> <reference_image> <output_video>\n";
        return 1;
    }

    std::string video_path = argv[1];
    std::string ref_img_path = argv[2];
    std::string output_path = argv[3];

    cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
    if (ref_image.empty()) {
        std::cerr << "Error: Could not read reference image.\n";
        return 1;
    }

    float ref_cdf_B[256], ref_cdf_G[256], ref_cdf_R[256];
    
    // TIME PRECOMPUTATION
    auto precomp_start = std::chrono::high_resolution_clock::now();
    SequentialMatcher::computeCDF(ref_image, ref_cdf_B, ref_cdf_G, ref_cdf_R);
    auto precomp_end = std::chrono::high_resolution_clock::now();
    double precomp_time = std::chrono::duration<double>(precomp_end - precomp_start).count();

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open target video.\n";
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    
    double total_read_time = 0;
    double total_write_time = 0;
    double total_compute_time = 0;
    int frame_count = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    while (true) {
        auto r_start = std::chrono::high_resolution_clock::now();
        bool success = cap.read(frame);
        auto r_end = std::chrono::high_resolution_clock::now();
        total_read_time += std::chrono::duration<double>(r_end - r_start).count();
        
        if (!success) break;

        auto c_start = std::chrono::high_resolution_clock::now();
        SequentialMatcher::matchHistogram(frame, ref_cdf_B, ref_cdf_G, ref_cdf_R);
        auto c_end = std::chrono::high_resolution_clock::now();
        total_compute_time += std::chrono::duration<double>(c_end - c_start).count();

        auto w_start = std::chrono::high_resolution_clock::now();
        writer.write(frame);
        auto w_end = std::chrono::high_resolution_clock::now();
        total_write_time += std::chrono::duration<double>(w_end - w_start).count();

        frame_count++;
        if (frame_count % 30 == 0 || frame_count == total_frames) {
            std::cout << "Processed " << frame_count << "/" << total_frames << " frames\r" << std::flush;
        }
    }
    std::cout << "\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n--- Sequential Baseline Stats ---\n";
    std::cout << "Total Frames: " << frame_count << "\n";
    std::cout << "Compute Time: " << std::fixed << std::setprecision(4) << total_compute_time << " s\n";
    std::cout << "Read Time:    " << std::fixed << std::setprecision(4) << total_read_time << " s\n";
    std::cout << "Write Time:   " << std::fixed << std::setprecision(4) << total_write_time << " s\n";
    std::cout << "Total Time:   " << std::fixed << std::setprecision(4) << total_time << " s\n";
    std::cout << "Throughput:   " << std::fixed << std::setprecision(2) << (frame_count / total_time) << " FPS\n";
    std::cout << "Overhead:     0.0000 s (Sequential)\n";

    return 0;
}
