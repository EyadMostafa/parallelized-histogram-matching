cmake_minimum_required(VERSION 3.15)
project(ParallelHistogramMatching C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

find_package(MPI COMPONENTS C)

add_executable(ParallelHistogramMatching 
    main.cpp
    histogram_matcher.cpp
)
target_link_libraries(ParallelHistogramMatching PRIVATE 
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

add_executable(SequentialHistogramMatching 
    sequential_main.cpp
    sequential_matcher.cpp
)
target_link_libraries(SequentialHistogramMatching PRIVATE 
    ${OpenCV_LIBS}
)

set(MPI_C_FOUND TRUE)
set(MPI_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/msmpi_sdk/PFiles/Microsoft SDKs/MPI/Include")
set(MPI_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/msmpi_sdk/PFiles/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

add_executable(HybridHistogramMatching 
    hybrid_main.cpp
    histogram_matcher.cpp
)
target_include_directories(HybridHistogramMatching PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(HybridHistogramMatching PRIVATE 
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    ${MPI_LIB_PATH}
)
