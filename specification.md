# **Technical Specification: Parallelized RGB Histogram Matching**

## **1\. Project Scope**

**Objective:** Develop a high-performance C++ video processing pipeline utilizing OpenMP to apply RGB Histogram Matching to an input video stream based on a single static reference image.

**Target Algorithm:** Frame-by-frame 3-channel (Blue, Green, Red) Histogram Matching (Specification).

**Primary Parallelization Framework:** OpenMP (Open Multi-Processing).

**Media I/O Framework:** OpenCV (specifically VideoCapture and VideoWriter).

## **2\. Hardware & Architecture Strategy**

**Target Hardware:** 13th Gen Intel(R) Core(TM) i7-13700H (14 Cores: 6 P-Cores, 8 E-Cores / 20 Logical Threads), 32GB RAM.

**Parallelization Level:** **Intra-Frame (Pixel-Level) Parallelism**.

* *Rationale:* Given the 32GB RAM and CPU cache limits, loading uncompressed video batches leads to cache thrashing. A frame-by-frame pipeline ensures the current working set easily fits into the CPU's fast L3 cache.  
  **Scheduling Strategy:** schedule(dynamic) or schedule(guided).  
* *Rationale:* The hybrid P-Core/E-Core architecture will cause thread imbalance with default static scheduling. Dynamic scheduling ensures P-Cores process more pixel chunks while E-Cores process fewer, eliminating implicit barrier wait times.

## **3\. Directory Structure**

To maintain clean separation between I/O logic and mathematical processing, the project will use a minimalist multi-file structure.

project\_root/  
│  
├── CMakeLists.txt              \# Build configuration for OpenCV and OpenMP  
├── main.cpp                    \# I/O, argument parsing, and video loop  
├── histogram\_matcher.h         \# Declarations for processing functions  
├── histogram\_matcher.cpp       \# OpenMP pragmas and core mathematical logic  
│  
├── input/                      \# Directory for test media  
│   ├── target\_video.mp4  
│   └── reference\_image.jpg  
│  
└── output/                     \# Directory for processed video

## **4\. Implementation Plan**

### **Phase 0: Initialization & I/O (Sequential)**

* **Action:** main.cpp parses arguments, opens the target video via cv::VideoCapture, and loads the reference image via cv::imread.  
* **Action:** Calculate the reference image's 3-channel (BGR) histogram and normalized Cumulative Distribution Function (CDF). This is performed *once* sequentially before the video loop begins.

### **Phase 1: Target Frame Histogram Calculation (Parallel)**

* **Action:** Inside the video frame loop, extract the current frame.  
* **OpenMP Strategy:** Iterate over the interleaved BGR pixel array.  
* **Optimization:** Avoid \#pragma omp atomic for every pixel, as 20 threads writing to a 256-bin array will cause massive contention.  
* **Thread-Local Reduction:** Each OpenMP thread will allocate its own private hist\_B\[256\], hist\_G\[256\], and hist\_R\[256\] arrays. Threads process their assigned pixel chunks into their private arrays. An OpenMP reduction (or sequential loop at the end of the parallel region) sums these private arrays into the global frame histograms.

### **Phase 2: CDF and LUT Generation (Sequential)**

* **Action:** A single master thread converts the global target frame histograms into CDFs.  
* **Action:** The master thread generates three independent Lookup Tables (LUT\_B, LUT\_G, LUT\_R) by matching the target frame's CDFs against the reference image's CDFs.  
* *Complexity:* ![][image1] operations. Parallelization overhead would exceed computation time; strictly sequential.

### **Phase 3: LUT Application (Embarrassingly Parallel)**

* **Action:** Apply the calculated LUTs back to the target frame's raw pixel data.  
* **OpenMP Strategy:** \#pragma omp parallel for schedule(dynamic).  
* **Logic:** Threads independently read a pixel's B, G, and R values, look up the new values in the respective LUTs, and overwrite the pixel in place. No synchronization or atomics are required since threads map 1:1 to unique memory addresses.

### **Phase 4: Output (Sequential)**

* **Action:** Pass the modified frame buffer to cv::VideoWriter.  
* **Action:** Loop back to Phase 1 until the video stream ends.

## **5\. Build System**

The project will use **CMake** to handle the complexities of linking both OpenCV and OpenMP cross-platform.

* find\_package(OpenCV REQUIRED) will locate the I/O libraries.  
* find\_package(OpenMP REQUIRED) and target\_link\_libraries(target OpenMP::OpenMP\_CXX) will securely link the threading flags.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAYCAYAAABEHYUrAAAD9UlEQVR4Xu1YO2tUQRS+iwqKL3yEjcnunbvLYhBUlFWDYGEhoogipjCQ6C+wskhaQVLYiZUE+0C0DVoEDNrZWCpikRW0UEIaI6hE/c7cM3fPPXfuZje7Ygo/mOzOd54zc+axCYIeoyD+rg/d2G4AdJp+p/rdIS9aHi8QhuENzaXRwolXJEmvQoIoiobr9foWzbeF0ITjoTFvDRocXdByDehNo41o/q9DzAHiL1er1d1NJg9shIG9gtEPfEZOwtxvfPYzl0K5XL5COpqv1Wq7YPeNbNn+ntYB30BFPBwcHCz19fXtqFQqp4wJP4I/oHVLpdI28J+dP7TjsSROnuyJFyZ+FIvF7ezgkZYR3IADTz0RT4k0mUKApEGbZcegf5T9JxwB/QbzScPgb0odAuIf4zh7qQ9/5335gJutRNF9yaVAtc6B5rXMAQlUOZFJyYO7hbYoOeYXsULvgySZAnEz5AOJjwo9WtlxfE6DP5udymZ+cjuhv0Tc0NDQTqk7MDCwnyfBDwhXWMETKoYrEUpO8hgQ2Y5JLubNL9a/6DgMqs7cgtBrkG/X98HwJCm6kGdHuoh1TvNUHqOcwIyWSdAMst6KoDcTR7MpOAscFAcheyA5JHCGfSQV1OZgyeYnfafVR7satFgY6L6UMRJVkN/JGfZAsSnMQiSarKwr7RZxU4D+M9LHgXbYceQP7TnaQhiaCZKrPWsn1MTVt4T+JjrMmLtECjo6+CnST5GYoa1spEskA+g8Yd1ktZDU5XZsCXQlsP1TyaPfQB6R6NPJRvtzmPp2+4RxjvJaQX+MuHLqYIzRzEtMg1sZE89YS7CeN2Dc0/ObBvRW0R6nSGkivnMse2qLsyI1qWL/T+nQ3kUQ5ZA6dCzSwU+TXqTuUnDXMk49gN07JDCheQL4PZpTgytwP1WWvsNOyLKDFXdry5U18apkTmuvUwXI5zHY64KiPWgPD9hPcvzU3c5c4pf7arDGDjb0XJe5eYFc9g3EgRIjOVWBR3bI65QB2XSorgBaERoky2+TPQ6sE0LFreQbR+D7HK44Xcb2wJR3r4OxB1SYPqAIGMQ+dp6ZIXCvSUYVoGWM3KuHVpP9ZhoGd5LVyF6Xpz2R5dng9i1eTzVLFKz/F0a9xhxMfPXMad6Cy9mWKoJ9wOfXMJ615KWTBxNfXb5HRTI43eQ+RYx+5j85uXsSSvC9TXL3Nv6idRzo9KaV13wLeKs6AyR7x3iei/8Kzedie/kHHSgS7B5L/xDo0MM6kOcfucyi3dV8zwDnIxV3LeVlkcf3BLFz3turXcZa29r8qx/vAqbtH++MtYeVj9D+W6YbD23CEwLb6Ij+t4xH7T82Dnq0PH8AO21VTeYUcIAAAAAASUVORK5CYII=>