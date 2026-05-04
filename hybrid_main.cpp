#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

#include "histogram_matcher.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) std::cerr << "Usage: mpiexec -n <cores> " << argv[0] << " <target_video> <reference_image> <output_video>\n";
        MPI_Finalize();
        return 1;
    }

    std::string video_path = argv[1];
    std::string ref_img_path = argv[2];
    std::string output_path = argv[3];

    float ref_cdf_B[256], ref_cdf_G[256], ref_cdf_R[256];
    
    // Rank 0 computes reference CDF and broadcasts to all
    if (rank == 0) {
        cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
        HistogramMatcher::computeCDF(ref_image, ref_cdf_B, ref_cdf_G, ref_cdf_R);
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
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = cap.get(cv::CAP_PROP_FPS);
        total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        frame_bytes = width * height * 3;
        writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    }

    MPI_Bcast(&frame_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    double compute_time = 0.0;
    double comm_time = 0.0;
    double read_time = 0.0;
    double write_time = 0.0;

    if (size == 1) {
        // Fallback for single node run
        if (rank == 0) {
            cv::Mat frame;
            int count = 0;
            while(cap.read(frame)) {
                HistogramMatcher::matchHistogram(frame, ref_cdf_B, ref_cdf_G, ref_cdf_R);
                writer.write(frame);
                count++;
                if (count % 30 == 0) std::cout << "Processed " << count << "\r" << std::flush;
            }
        }
    } else {
        if (rank == 0) {
            // MASTER LOGIC
            int frames_read = 0;
            int active_workers = 0;
            std::map<int, cv::Mat> frame_buffer;
            int next_write_idx = 0;
            
            // Seed workers
            for (int i = 1; i < size && frames_read < total_frames; ++i) {
                cv::Mat frame;
                double r_t0 = MPI_Wtime();
                bool success = cap.read(frame);
                read_time += (MPI_Wtime() - r_t0);

                if (success) {
                    double t0 = MPI_Wtime();
                    MPI_Send(&frames_read, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(frame.data, frame_bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                    comm_time += (MPI_Wtime() - t0);
                    active_workers++;
                    frames_read++;
                }
            }

            cv::Mat recv_frame(height, width, CV_8UC3);
            int written_frames = 0;
            
            while (active_workers > 0) {
                MPI_Status status;
                int received_idx;
                
                double t0 = MPI_Wtime();
                MPI_Recv(&received_idx, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                MPI_Recv(recv_frame.data, frame_bytes, MPI_BYTE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                comm_time += (MPI_Wtime() - t0);
                active_workers--;
                
                frame_buffer[received_idx] = recv_frame.clone();
                
                // Write ordered frames
                while (frame_buffer.find(next_write_idx) != frame_buffer.end()) {
                    double w_t0 = MPI_Wtime();
                    writer.write(frame_buffer[next_write_idx]);
                    write_time += (MPI_Wtime() - w_t0);

                    frame_buffer.erase(next_write_idx);
                    next_write_idx++;
                    written_frames++;
                    if (written_frames % 30 == 0 || written_frames == total_frames) {
                        std::cout << "Processed " << written_frames << "/" << total_frames << " frames\r" << std::flush;
                    }
                }

                if (frames_read < total_frames) {
                    cv::Mat next_frame;
                    double r_t0 = MPI_Wtime();
                    bool success = cap.read(next_frame);
                    read_time += (MPI_Wtime() - r_t0);

                    if (success) {
                        t0 = MPI_Wtime();
                        MPI_Send(&frames_read, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
                        MPI_Send(next_frame.data, frame_bytes, MPI_BYTE, source, 0, MPI_COMM_WORLD);
                        comm_time += (MPI_Wtime() - t0);
                        active_workers++;
                        frames_read++;
                    }
                }
            }

            // Terminate workers
            for (int i = 1; i < size; ++i) {
                int term_signal = -1;
                MPI_Send(&term_signal, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            }
            std::cout << "\n";
        } else {
            // WORKER LOGIC
            cv::Mat frame(height, width, CV_8UC3);
            while (true) {
                MPI_Status status;
                int frame_idx;
                
                double t0 = MPI_Wtime();
                MPI_Recv(&frame_idx, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == 2) break; // Terminate
                
                MPI_Recv(frame.data, frame_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                comm_time += (MPI_Wtime() - t0);
                
                t0 = MPI_Wtime();
                // Applies OpenMP thread-level parallelism inside the node
                HistogramMatcher::matchHistogram(frame, ref_cdf_B, ref_cdf_G, ref_cdf_R);
                compute_time += (MPI_Wtime() - t0);

                t0 = MPI_Wtime();
                MPI_Send(&frame_idx, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(frame.data, frame_bytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
                comm_time += (MPI_Wtime() - t0);
            }
        }
    }

    double total_time = MPI_Wtime() - start_time;
    // (Master already has comm_time, Workers have compute_time and comm_time)

    // Aggregate compute time from all nodes
    double global_compute_time = 0;
    MPI_Reduce(&compute_time, &global_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Barrier to ensure print formatting is somewhat clean
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n--- MPI + OpenMP Hybrid Execution Stats ---\n";
        std::cout << "Nodes/Processes: " << size << "\n";
        std::cout << "Compute Time: " << std::fixed << std::setprecision(4) << global_compute_time << " s (Total Cluster)\n";
        std::cout << "Read Time:    " << std::fixed << std::setprecision(4) << read_time << " s\n";
        std::cout << "Write Time:   " << std::fixed << std::setprecision(4) << write_time << " s\n";
        std::cout << "Comm Overhead: " << std::fixed << std::setprecision(4) << comm_time << " s (Master)\n";
        std::cout << "Total Time:    " << std::fixed << std::setprecision(4) << total_time << " s\n";
    } else {
        std::cout << "Worker " << rank << " - Compute: " << std::fixed << std::setprecision(4) << compute_time 
                  << " s, Comm: " << comm_time << " s\n";
    }

    MPI_Finalize();
    return 0;
}
