#include <iostream>
#include <thread>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "histogram_matcher.h"
#include "thread_safe_queue.h"

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
        std::cerr << "Example: " << argv[0] << " input/target.mp4 input/ref.jpg output/out.mp4\n";
        return 1;
    }

    std::string video_path   = argv[1];
    std::string ref_img_path = argv[2];
    std::string output_path  = argv[3];

    // ── Level 0: Precompute reference CDF (once, before any threading) ────────
    cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
    if (ref_image.empty()) { std::cerr << "Error: Could not read reference image at " << ref_img_path << "\n"; return 1; }

    float ref_cdf_B[256], ref_cdf_G[256], ref_cdf_R[256];
    auto t_precomp = Clock::now();
    HistogramMatcher::computeCDF(ref_image, ref_cdf_B, ref_cdf_G, ref_cdf_R);
    double precomp_time = elapsed(t_precomp);

    // ── Video I/O setup ───────────────────────────────────────────────────────
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open target video at " << video_path << "\n"; return 1; }

    int    width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps          = cap.get(cv::CAP_PROP_FPS);
    int    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) { std::cerr << "Error: Could not open output video for writing at " << output_path << "\n"; return 1; }

    // Bounded queues: 15 frames ≈ ~90 MB for 1080p; large enough to absorb I/O jitter
    ThreadSafeQueue<cv::Mat> read_queue(15);
    ThreadSafeQueue<cv::Mat> write_queue(15);

    double total_read_time       = 0;
    double total_write_time      = 0;
    double total_cdf_time        = 0;
    double total_lut_gen_time    = 0;
    double total_lut_apply_time  = 0;
    // Time the compute thread spends blocked on queue operations (true sync overhead)
    double total_read_wait_time  = 0;
    double total_write_wait_time = 0;

    auto start_time = Clock::now();

    // ── Phase 1: Producer Thread — Read frames from disk ─────────────────────
    std::thread reader_thread([&]() {
        cv::Mat frame;
        while (true) {
            auto t = Clock::now();
            bool ok = cap.read(frame);
            total_read_time += elapsed(t);
            if (!ok) break;
            read_queue.push(frame.clone());
        }
        read_queue.set_finished();
    });

    // ── Phase 2: Consumer Thread — Encode and write frames to disk ────────────
    std::thread writer_thread([&]() {
        while (true) {
            auto opt_frame = write_queue.pop();
            if (!opt_frame) break;
            auto t = Clock::now();
            writer.write(*opt_frame);
            total_write_time += elapsed(t);
        }
    });

    // ── Phase 3: Compute Thread (main thread) — OpenMP inside ────────────────
    int frame_count = 0;

    float src_cdf_B[256], src_cdf_G[256], src_cdf_R[256];
    uchar lut_B[256],     lut_G[256],     lut_R[256];

    while (true) {
        // Time spent blocked waiting for reader to produce a frame
        auto t_rwait = Clock::now();
        auto opt_frame = read_queue.pop();
        total_read_wait_time += elapsed(t_rwait);
        if (!opt_frame) {
            write_queue.set_finished();
            break;
        }

        cv::Mat frame = *opt_frame;

        // Level 2a: computeCDF
        auto t_cdf = Clock::now();
        HistogramMatcher::computeCDF(frame, src_cdf_B, src_cdf_G, src_cdf_R);
        total_cdf_time += elapsed(t_cdf);

        // Level 2b: generateLUTs
        auto t_gen = Clock::now();
        HistogramMatcher::generateLUTs(src_cdf_B, src_cdf_G, src_cdf_R,
                                       ref_cdf_B,  ref_cdf_G,  ref_cdf_R,
                                       lut_B, lut_G, lut_R);
        total_lut_gen_time += elapsed(t_gen);

        // Level 2c: applyLUT
        auto t_apply = Clock::now();
        HistogramMatcher::applyLUT(frame, lut_B, lut_G, lut_R);
        total_lut_apply_time += elapsed(t_apply);

        // Time spent blocked waiting for the writer queue to have space
        auto t_wwait = Clock::now();
        write_queue.push(frame);
        total_write_wait_time += elapsed(t_wwait);

        frame_count++;
        if (frame_count % 30 == 0 || frame_count == total_frames)
            std::cout << "Processed " << frame_count << "/" << total_frames << " frames\r" << std::flush;
    }
    std::cout << "\n";

    reader_thread.join();
    writer_thread.join();

    double total_time         = elapsed(start_time);
    double total_compute_time = total_cdf_time + total_lut_gen_time + total_lut_apply_time;
    // True sync overhead = time compute thread was idle, blocked on queue operations
    double sync_overhead      = total_read_wait_time + total_write_wait_time - total_compute_time;
    if (sync_overhead < 0) sync_overhead = 0; // clamp: if compute > wait, no stall overhead

    // ── Report ────────────────────────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "=== OpenMP Parallel Pipeline — Per-Stage Latency ===\n";
    std::cout << " Frames : " << frame_count << "  (" << width << "x" << height << " @ " << fps << " fps)\n";
    std::cout << " -----------------------------------------------\n";
    print_row("[L0] Precomp (ref CDF)",    precomp_time,        frame_count);
    print_row("[L1] Read + Decode",        total_read_time,     frame_count);
    print_row("[L2] Compute (total)",      total_compute_time,  frame_count);
    print_row("computeCDF",               total_cdf_time,      frame_count, true);
    print_row("generateLUTs",             total_lut_gen_time,  frame_count, true);
    print_row("applyLUT",                 total_lut_apply_time,frame_count, true);
    print_row("[L3] Write + Encode",       total_write_time,    frame_count);
    print_row("[L4] Queue Block (Sync)",   sync_overhead,       frame_count);
    std::cout << "      (read_wait=" << std::fixed << std::setprecision(4) << total_read_wait_time
              << " s, write_wait=" << total_write_wait_time << " s)\n";
    std::cout << " -----------------------------------------------\n";
    std::cout << " Total Time  : " << std::fixed << std::setprecision(4) << total_time << " s\n";
    std::cout << " Throughput  : " << std::fixed << std::setprecision(2) << (frame_count / total_time) << " FPS\n";

    return 0;
}
