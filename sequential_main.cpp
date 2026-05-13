#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "sequential_matcher.h"

using Clock = std::chrono::high_resolution_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

static void print_row(const std::string& label, double total_s, int frames, bool is_sub = false) {
    double ms_per = (frames > 0) ? (total_s / frames * 1000.0) : 0.0;
    std::string prefix = is_sub ? "     " : " ";
    std::cout << prefix
              << std::left  << std::setw(26) << label
              << std::right << std::setw(9)  << std::fixed << std::setprecision(4) << total_s << " s"
              << std::setw(10) << std::fixed << std::setprecision(3) << ms_per << " ms/frame\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_video> <reference_image> <output_video>\n";
        return 1;
    }

    std::string video_path   = argv[1];
    std::string ref_img_path = argv[2];
    std::string output_path  = argv[3];

    // ── Level 0: One-time precomputation (reference CDF) ─────────────────────
    cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
    if (ref_image.empty()) { std::cerr << "Error: Could not read reference image.\n"; return 1; }

    float ref_cdf_B[256], ref_cdf_G[256], ref_cdf_R[256];
    auto t_precomp = Clock::now();
    SequentialMatcher::computeCDF(ref_image, ref_cdf_B, ref_cdf_G, ref_cdf_R);
    double precomp_time = elapsed(t_precomp);

    // ── Video I/O setup ───────────────────────────────────────────────────────
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open target video.\n"; return 1; }

    int    width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps          = cap.get(cv::CAP_PROP_FPS);
    int    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) { std::cerr << "Error: Could not open output video for writing.\n"; return 1; }

    // ── Per-stage accumulators ────────────────────────────────────────────────
    double total_read_time      = 0;
    double total_cdf_time       = 0;
    double total_lut_gen_time   = 0;
    double total_lut_apply_time = 0;
    double total_write_time     = 0;
    int    frame_count          = 0;

    // Reuse buffers across frames (avoids repeated stack allocation)
    float src_cdf_B[256], src_cdf_G[256], src_cdf_R[256];
    uchar lut_B[256],     lut_G[256],     lut_R[256];

    auto total_start = Clock::now();
    cv::Mat frame;

    while (true) {
        // ── Level 1: Read + Decode ────────────────────────────────────────────
        auto t_read = Clock::now();
        bool ok = cap.read(frame);
        total_read_time += elapsed(t_read);
        if (!ok) break;

        // ── Level 2a: computeCDF ──────────────────────────────────────────────
        auto t_cdf = Clock::now();
        SequentialMatcher::computeCDF(frame, src_cdf_B, src_cdf_G, src_cdf_R);
        total_cdf_time += elapsed(t_cdf);

        // ── Level 2b: generateLUTs ────────────────────────────────────────────
        auto t_gen = Clock::now();
        SequentialMatcher::generateLUTs(src_cdf_B, src_cdf_G, src_cdf_R,
                                        ref_cdf_B,  ref_cdf_G,  ref_cdf_R,
                                        lut_B, lut_G, lut_R);
        total_lut_gen_time += elapsed(t_gen);

        // ── Level 2c: applyLUT ────────────────────────────────────────────────
        auto t_apply = Clock::now();
        SequentialMatcher::applyLUT(frame, lut_B, lut_G, lut_R);
        total_lut_apply_time += elapsed(t_apply);

        // ── Level 3: Encode + Write ───────────────────────────────────────────
        auto t_write = Clock::now();
        writer.write(frame);
        total_write_time += elapsed(t_write);

        frame_count++;
        if (frame_count % 30 == 0 || frame_count == total_frames)
            std::cout << "Processed " << frame_count << "/" << total_frames << " frames\r" << std::flush;
    }
    std::cout << "\n";

    double total_time         = elapsed(total_start);
    double total_compute_time = total_cdf_time + total_lut_gen_time + total_lut_apply_time;

    // ── Report ────────────────────────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "=== Sequential Baseline — Per-Stage Latency ===\n";
    std::cout << " Frames : " << frame_count << "  (" << width << "x" << height << " @ " << fps << " fps)\n";
    std::cout << " -----------------------------------------------\n";
    print_row("[L0] Precomp (ref CDF)",    precomp_time,       frame_count);
    print_row("[L1] Read + Decode",        total_read_time,    frame_count);
    print_row("[L2] Compute (total)",      total_compute_time, frame_count);
    print_row("computeCDF",               total_cdf_time,     frame_count, true);
    print_row("generateLUTs",             total_lut_gen_time, frame_count, true);
    print_row("applyLUT",                 total_lut_apply_time, frame_count, true);
    print_row("[L3] Write + Encode",       total_write_time,   frame_count);
    print_row("[L4] Sync/Overhead",        0.0,                frame_count);
    std::cout << " -----------------------------------------------\n";
    std::cout << " Total Time  : " << std::fixed << std::setprecision(4) << total_time << " s\n";
    std::cout << " Throughput  : " << std::fixed << std::setprecision(2) << (frame_count / total_time) << " FPS\n";

    return 0;
}
