#pragma once

#include <vector>
#include <string>

constexpr int IMAGE_HEIGHT = 1024;
constexpr int IMAGE_WIDTH = 1024;
constexpr int FILTER_HEIGHT = 19;
constexpr int FILTER_WIDTH = 19;

std::vector<int> generateRandomImage();

class Timer {
 public:
  Timer(std::string task_name);

  ~Timer();

 private:
  std::string task_name;
  std::chrono::time_point<std::chrono::steady_clock> start;
};

void convolution_cpu(const std::vector<int> &image,
					 const std::vector<int> &filter,
					 std::vector<int> &result);

void convolution_cpu_cache(const std::vector<int> &image,
						   const std::vector<int> &filter,
						   std::vector<int> &result);

void convolution_cpu_cache_optimized(const std::vector<int> &image,
									 const std::vector<int> &filter,
									 std::vector<int> &result);

void convolution_simd_4(const std::vector<int> &image,
						const std::vector<int> &filter,
						std::vector<int> &result);

void convolution_simd_16(const std::vector<int> &image,
						 const std::vector<int> &filter,
						 std::vector<int> &result);

void convolution_opencl(const std::vector<int> &image,
						const std::vector<int> &filter,
						std::vector<int> &result);

void convolution_opencl_parallel(const std::vector<int> &image,
								 const std::vector<int> &filter,
								 std::vector<int> &result);
