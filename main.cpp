#include <iostream>
#include <thread>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "histogram_matcher.h"
#include "thread_safe_queue.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_video> <reference_image> <output_video>\n";
        std::cerr << "Example: " << argv[0] << " input/target_video.mp4 input/reference_image.jpg output/processed_video.mp4\n";
        return 1;
    }

    std::string video_path = argv[1];
    std::string ref_img_path = argv[2];
    std::string output_path = argv[3];

    // Load reference image
    cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
    if (ref_image.empty()) {
        std::cerr << "Error: Could not read reference image at " << ref_img_path << "\n";
        return 1;
    }

    std::cout << "Precomputing reference CDF...\n";
    // Precompute reference CDF
    float ref_cdf_B[256], ref_cdf_G[256], ref_cdf_R[256];
    HistogramMatcher::computeCDF(ref_image, ref_cdf_B, ref_cdf_G, ref_cdf_R);

    // Open video capture
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open target video at " << video_path << "\n";
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Video properties: " << width << "x" << height << " @ " << fps << " fps (" << total_frames << " frames)\n";

    // Open video writer
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing at " << output_path << "\n";
        return 1;
    }

    // Queues for pipelining
    // Size 15 is large enough to hide I/O jitter but small enough to keep memory usage low (e.g. 15 frames of 1080p is ~90MB)
    ThreadSafeQueue<cv::Mat> read_queue(15);
    ThreadSafeQueue<cv::Mat> write_queue(15);

    double total_read_time = 0;
    double total_write_time = 0;
    double total_compute_time = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------
    // Phase 1: Producer Thread (Read frames from disk)
    // ---------------------------------------------------------
    std::thread reader_thread([&]() {
        cv::Mat frame;
        while (true) {
            auto r_start = std::chrono::high_resolution_clock::now();
            bool success = cap.read(frame);
            auto r_end = std::chrono::high_resolution_clock::now();
            total_read_time += std::chrono::duration<double>(r_end - r_start).count();
            
            if (!success) break;
            read_queue.push(frame.clone()); 
        }
        read_queue.set_finished();
    });

    // ---------------------------------------------------------
    // Phase 2: Consumer Thread (Write frames to disk)
    // ---------------------------------------------------------
    std::thread writer_thread([&]() {
        while (true) {
            auto opt_frame = write_queue.pop();
            if (!opt_frame) break;
            
            auto w_start = std::chrono::high_resolution_clock::now();
            writer.write(*opt_frame);
            auto w_end = std::chrono::high_resolution_clock::now();
            total_write_time += std::chrono::duration<double>(w_end - w_start).count();
        }
    });

    // ---------------------------------------------------------
    // Phase 3: Compute Thread (Main thread running OpenMP)
    // ---------------------------------------------------------
    int frame_count = 0;
    while (true) {
        auto opt_frame = read_queue.pop();
        if (!opt_frame) {
            write_queue.set_finished();
            break;
        }

        cv::Mat frame = *opt_frame;
        
        auto c_start = std::chrono::high_resolution_clock::now();
        // This invokes OpenMP parallel regions inside
        HistogramMatcher::matchHistogram(frame, ref_cdf_B, ref_cdf_G, ref_cdf_R);
        auto c_end = std::chrono::high_resolution_clock::now();
        total_compute_time += std::chrono::duration<double>(c_end - c_start).count();
        
        write_queue.push(frame);

        frame_count++;
        if (frame_count % 30 == 0 || frame_count == total_frames) {
            std::cout << "Processed " << frame_count << "/" << total_frames << " frames\r" << std::flush;
        }
    }
    std::cout << "\n";

    // Wait for I/O threads to finish
    reader_thread.join();
    writer_thread.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "\n--- OpenMP Parallel Execution Stats ---\n";
    std::cout << "Total Frames: " << frame_count << "\n";
    std::cout << "Compute Time: " << std::fixed << std::setprecision(4) << total_compute_time << " s\n";
    std::cout << "Read Time:    " << std::fixed << std::setprecision(4) << total_read_time << " s\n";
    std::cout << "Write Time:   " << std::fixed << std::setprecision(4) << total_write_time << " s\n";
    std::cout << "Total Time:   " << std::fixed << std::setprecision(4) << diff.count() << " s\n";
    std::cout << "Throughput:   " << std::fixed << std::setprecision(2) << (frame_count / diff.count()) << " FPS\n";
    std::cout << "Overhead:     " << std::fixed << std::setprecision(4) << (diff.count() - (total_compute_time / std::thread::hardware_concurrency())) << " s (Queue/Sync)\n";

    return 0;
}
