#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "histogram_matcher.h"

static void print_row(const std::string& label, double total_s, int frames, bool is_sub = false) {
    double ms_per = (frames > 0) ? (total_s / frames * 1000.0) : 0.0;
    std::string prefix = is_sub ? "     " : " ";
    std::cout << prefix
              << std::left  << std::setw(28) << label
              << std::right << std::setw(9)  << std::fixed << std::setprecision(4) << total_s << " s"
              << std::setw(10) << std::fixed << std::setprecision(3) << ms_per << " ms/frame\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) std::cerr << "Usage: mpiexec -n <cores> " << argv[0]
                                 << " <target_video> <reference_image> <output_video>\n";
        MPI_Finalize();
        return 1;
    }

    std::string video_path   = argv[1];
    std::string ref_img_path = argv[2];
    std::string output_path  = argv[3];

    // ── Level 0: Rank 0 computes ref CDF, broadcasts to all workers ───────────
    float ref_cdf_B[256], ref_cdf_G[256], ref_cdf_R[256];
    double precomp_time = 0.0;

    if (rank == 0) {
        cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
        if (ref_image.empty()) {
            std::cerr << "Error: Could not read reference image.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double t0 = MPI_Wtime();
        HistogramMatcher::computeCDF(ref_image, ref_cdf_B, ref_cdf_G, ref_cdf_R);
        precomp_time = MPI_Wtime() - t0;
    }

    MPI_Bcast(ref_cdf_B, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ref_cdf_G, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ref_cdf_R, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int frame_bytes = 0, width = 0, height = 0, total_frames = 0;
    double fps = 0;
    cv::VideoCapture cap;
    cv::VideoWriter writer;

    if (rank == 0) {
        cap.open(video_path);
        if (!cap.isOpened()) { std::cerr << "Error: Could not open video.\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = cap.get(cv::CAP_PROP_FPS);
        total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        frame_bytes = width * height * 3;
        writer.open(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    }

    // Broadcast video geometry so workers can pre-allocate frame buffers
    MPI_Bcast(&frame_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_frames, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    // Per-stage timers (workers populate compute stages; master populates I/O)
    double read_time        = 0.0;
    double write_time       = 0.0;
    double comm_time        = 0.0;
    double compute_cdf      = 0.0;
    double compute_lut_gen  = 0.0;
    double compute_lut_apply= 0.0;

    // ── Single-process fallback ───────────────────────────────────────────────
    if (size == 1) {
        if (rank == 0) {
            float src_cdf_B[256], src_cdf_G[256], src_cdf_R[256];
            uchar lut_B[256],     lut_G[256],     lut_R[256];
            cv::Mat frame;
            int count = 0;
            while (true) {
                double t0;
                t0 = MPI_Wtime(); bool ok = cap.read(frame); 
                read_time += MPI_Wtime()-t0;
                if (!ok) break;
                t0 = MPI_Wtime();
                HistogramMatcher::computeCDF(frame, src_cdf_B, src_cdf_G, src_cdf_R); 
                compute_cdf += MPI_Wtime()-t0;
                t0 = MPI_Wtime();
                HistogramMatcher::generateLUTs(src_cdf_B,src_cdf_G,src_cdf_R,ref_cdf_B,ref_cdf_G,ref_cdf_R,lut_B,lut_G,lut_R);
                compute_lut_gen += MPI_Wtime()-t0;
                t0 = MPI_Wtime(); 
                HistogramMatcher::applyLUT(frame,lut_B,lut_G,lut_R); 
                compute_lut_apply += MPI_Wtime()-t0;
                t0 = MPI_Wtime();
                writer.write(frame);
                write_time += MPI_Wtime()-t0;
                count++;
                if (count % 30 == 0) std::cout << "Processed " << count << "\r" << std::flush;
            }
            total_frames = count;
        }
    } else {
        // ── MASTER (rank 0) ───────────────────────────────────────────────────
        if (rank == 0) {
            int frames_read = 0;
            int active_workers = 0;
            std::map<int, cv::Mat> frame_buffer;
            int next_write_idx = 0;
            int written_frames = 0;

            // Seed each worker with its first frame
            for (int i = 1; i < size && frames_read < total_frames; ++i) {
                cv::Mat frame;
                double t0 = MPI_Wtime();
                bool ok = cap.read(frame); 
                read_time += MPI_Wtime()-t0;
                if (!ok) break;
                t0 = MPI_Wtime();
                MPI_Send(&frames_read, 1, MPI_INT,  i, 0, MPI_COMM_WORLD);
                MPI_Send(frame.data, frame_bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                comm_time += MPI_Wtime()-t0;
                active_workers++;
                frames_read++;
            }

            cv::Mat recv_frame(height, width, CV_8UC3);

            while (active_workers > 0) {
                MPI_Status status;
                int received_idx;
                // Receive processed frame from any worker
                double t0 = MPI_Wtime();
                MPI_Recv(&received_idx, 1, MPI_INT,  MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                MPI_Recv(recv_frame.data, frame_bytes, MPI_BYTE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                comm_time += MPI_Wtime()-t0;
                active_workers--;

                frame_buffer[received_idx] = recv_frame.clone();

                // Flush any contiguous run of ordered frames to disk
                while (frame_buffer.count(next_write_idx)) {
                    t0 = MPI_Wtime(); 
                    writer.write(frame_buffer[next_write_idx]);
                    write_time += MPI_Wtime()-t0;
                    frame_buffer.erase(next_write_idx);
                    next_write_idx++;
                    written_frames++;
                    if (written_frames % 30 == 0 || written_frames == total_frames)
                        std::cout << "Processed " << written_frames << "/" << total_frames << " frames\r" << std::flush;
                }

                // Re-feed this worker with the next unprocessed frame
                if (frames_read < total_frames) {
                    cv::Mat next_frame;
                    t0 = MPI_Wtime(); bool ok = cap.read(next_frame); read_time += MPI_Wtime()-t0;
                    if (ok) {
                        t0 = MPI_Wtime();
                        MPI_Send(&frames_read, 1, MPI_INT,  source, 0, MPI_COMM_WORLD);
                        MPI_Send(next_frame.data, frame_bytes, MPI_BYTE, source, 0, MPI_COMM_WORLD);
                        comm_time += MPI_Wtime()-t0;
                        active_workers++;
                        frames_read++;
                    }
                }
            }

            // Send termination signal (tag 2) to all workers
            for (int i = 1; i < size; ++i) {
                int term = -1;
                MPI_Send(&term, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            }
            std::cout << "\n";

        } else {
            // ── WORKER (rank > 0) ─────────────────────────────────────────────
            cv::Mat frame(height, width, CV_8UC3);
            float src_cdf_B[256], src_cdf_G[256], src_cdf_R[256];
            uchar lut_B[256], lut_G[256], lut_R[256];

            while (true) {
                MPI_Status status;
                int frame_idx;
                // Receive frame index (or termination tag 2)
                double t0 = MPI_Wtime();
                MPI_Recv(&frame_idx, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == 2) break;
                MPI_Recv(frame.data, frame_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                comm_time += MPI_Wtime()-t0;

                // Level 2a: computeCDF
                t0 = MPI_Wtime();
                HistogramMatcher::computeCDF(frame, src_cdf_B, src_cdf_G, src_cdf_R);
                compute_cdf += MPI_Wtime()-t0;

                // Level 2b: generateLUTs
                t0 = MPI_Wtime();
                HistogramMatcher::generateLUTs(src_cdf_B, src_cdf_G, src_cdf_R,
                                               ref_cdf_B,  ref_cdf_G,  ref_cdf_R,
                                               lut_B, lut_G, lut_R);
                compute_lut_gen += MPI_Wtime()-t0;

                // Level 2c: applyLUT
                t0 = MPI_Wtime();
                HistogramMatcher::applyLUT(frame, lut_B, lut_G, lut_R);
                compute_lut_apply += MPI_Wtime()-t0;

                // Return processed frame to master
                t0 = MPI_Wtime();
                MPI_Send(&frame_idx, 1, MPI_INT,  0, 1, MPI_COMM_WORLD);
                MPI_Send(frame.data, frame_bytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
                comm_time += MPI_Wtime()-t0;
            }
        }
    }

    double total_time = MPI_Wtime() - start_time;

    // ── Aggregate per-stage compute times from all workers ────────────────────
    double global_cdf = 0, global_lut_gen = 0, global_lut_apply = 0;
    double global_comm = 0;
    MPI_Reduce(&compute_cdf, &global_cdf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_lut_gen, &global_lut_gen, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_lut_apply, &global_lut_apply, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &global_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Per-worker console output (non-master) ────────────────────────────────
    if (rank != 0) {
        double w_compute = compute_cdf + compute_lut_gen + compute_lut_apply;
        std::cout << "  Worker " << rank
                  << " | Compute: " << std::fixed << std::setprecision(4) << w_compute
                  << " s (CDF=" << compute_cdf << " LUTgen=" << compute_lut_gen
                  << " LUTapply=" << compute_lut_apply << ")"
                  << " | Comm: " << comm_time << " s\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Master summary report ─────────────────────────────────────────────────
    if (rank == 0) {
        int n_workers       = std::max(1, size - 1);
        double avg_cdf      = global_cdf       / n_workers;
        double avg_lut_gen  = global_lut_gen   / n_workers;
        double avg_lut_apply= global_lut_apply / n_workers;
        double avg_compute  = avg_cdf + avg_lut_gen + avg_lut_apply;
        // Master-side comm is already in comm_time (not summed to global_comm)
        double master_comm  = comm_time;

        std::cout << "\n";
        std::cout << "=== MPI+OpenMP Hybrid — Per-Stage Latency ===\n";
        std::cout << " Processes : " << size << "  (" << (size-1) << " workers)\n";
        std::cout << " Frames    : " << total_frames << "  (" << width << "x" << height << " @ " << fps << " fps)\n";
        std::cout << " -------------------------------------------------\n";
        print_row("[L0] Precomp (ref CDF)",     precomp_time, total_frames);
        print_row("[L1] Read + Decode",          read_time,    total_frames);
        print_row("[L2] Compute (avg/worker)",   avg_compute,  total_frames);
        print_row("computeCDF",                 avg_cdf,      total_frames, true);
        print_row("generateLUTs",               avg_lut_gen,  total_frames, true);
        print_row("applyLUT",                   avg_lut_apply,total_frames, true);
        print_row("[L3] Write + Encode",         write_time,   total_frames);
        print_row("[L4] Comm Overhead (Master)", master_comm,  total_frames);
        std::cout << " -------------------------------------------------\n";
        std::cout << " Total Time  : " << std::fixed << std::setprecision(4) << total_time << " s\n";
        std::cout << " Throughput  : " << std::fixed << std::setprecision(2)
                  << (total_frames / total_time) << " FPS\n";
    }

    MPI_Finalize();
    return 0;
}
