#include "convolutions.h"

#include <jni.h>
#include <string>
#include <android/log.h>

#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <arm_neon.h>

#define  LOG_TAG    "NDK_TEST"
#define  LOGUNK(...)  __android_log_print(ANDROID_LOG_UNKNOWN,LOG_TAG,__VA_ARGS__)
#define  LOGDEF(...)  __android_log_print(ANDROID_LOG_DEFAULT,LOG_TAG,__VA_ARGS__)
#define  LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGF(...)  __android_log_print(ANDROID_FATAL_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGS(...)  __android_log_print(ANDROID_SILENT_ERROR,LOG_TAG,__VA_ARGS__)

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <fmt/format.h>
#include <cassert>

std::vector<int> generateRandomImage() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  std::vector<int> image(IMAGE_HEIGHT * IMAGE_WIDTH);
  for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
	image[i] = dis(gen);
  }

  return image;
}

Timer::Timer(std::string task_name) : task_name(std::move(task_name)) {
  start = std::chrono::steady_clock::now();
}

Timer::~Timer() {
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  long long elapsed_microseconds = elapsed.count();

  std::cout << "hello" << std::endl;
  auto str = fmt::format("{} took {} microseconds", task_name, elapsed_microseconds);
  __android_log_print(ANDROID_LOG_DEBUG, "Timer", "%s", str.c_str());
}

void convolution_cpu(const std::vector<int>& image,
					 const std::vector<int>& filter,
					 std::vector<int>& result) {
  for (int i = 0; i < IMAGE_HEIGHT - FILTER_HEIGHT + 1; ++i) {
	for (int j = 0; j < IMAGE_WIDTH - FILTER_WIDTH + 1; ++j) {
	  int sum = 0;
	  for (int fi = 0; fi < FILTER_HEIGHT; ++fi) {
		for (int fj = 0; fj < FILTER_WIDTH; ++fj) {
		  sum += image[(i + fi) * IMAGE_WIDTH + j + fj] * filter[fi * FILTER_WIDTH + fj];
		}
	  }
	  result[i * (IMAGE_WIDTH - FILTER_WIDTH + 1) + j] = sum;
	}
  }
}

void convolution_cpu_cache(const std::vector<int>& image,
						   const std::vector<int>& filter,
						   std::vector<int>& result) {
  int result_width = IMAGE_WIDTH - FILTER_WIDTH + 1;
  int result_height = IMAGE_HEIGHT - FILTER_HEIGHT + 1;

  // 이미 계산된 부분을 저장하기 위한 캐시
  std::vector<int> cache(result_height * result_width, 0);

  for (int i = 0; i < result_height; ++i) {
	for (int j = 0; j < result_width; ++j) {
	  // 이미 계산된 부분이 있는 경우, 캐시된 값을 사용합니다.
	  if (cache[i * result_width + j] != 0) {
		result[i * result_width + j] = cache[i * result_width + j];
		continue;
	  }

	  int sum = 0;

	  // 필터와의 합을 계산합니다.
	  for (int fi = 0; fi < FILTER_HEIGHT; ++fi) {
		for (int fj = 0; fj < FILTER_WIDTH; ++fj) {
		  sum += image[(i + fi) * IMAGE_WIDTH + j + fj] * filter[fi * FILTER_WIDTH + fj];
		}
	  }

	  // 계산된 결과를 캐시에 저장합니다.
	  cache[i * result_width + j] = sum;
	  result[i * result_width + j] = sum;
	}
  }
}

void convolution_cpu_cache_optimized(const std::vector<int>& image,
									 const std::vector<int>& filter,
									 std::vector<int>& result) {
  int result_width = IMAGE_WIDTH - FILTER_WIDTH + 1;
  int result_height = IMAGE_HEIGHT - FILTER_HEIGHT + 1;

  // 이미 계산된 부분을 저장하기 위한 캐시
  std::vector<int> cache(result_height * result_width, 0);

  for (int i = 0; i < result_height; ++i) {
	for (int j = 0; j < result_width; ++j) {
	  // 이미 계산된 부분이 있는 경우, 캐시된 값을 사용합니다.
	  if (cache[i * result_width + j] != 0) {
		result[i * result_width + j] = cache[i * result_width + j];
		continue;
	  }

	  int sum = 0;

	  // 필터와의 합을 계산합니다.
	  for (int fi = 0; fi < FILTER_HEIGHT; ++fi) {
		for (int fj = 0; fj < FILTER_WIDTH; ++fj) {
		  // 이미지와 필터의 각 픽셀의 곱셈 및 누적을 한 번에 처리합니다.
		  sum += image[(i + fi) * IMAGE_WIDTH + j + fj] * filter[fi * FILTER_WIDTH + fj];
		}
	  }

	  // 계산된 결과를 캐시에 저장합니다.
	  cache[i * result_width + j] = sum;
	  result[i * result_width + j] = sum;
	}
  }
}

void convolution_simd_4(const std::vector<int>& image,
						const std::vector<int>& filter,
						std::vector<int>& result) {
  for (int i = 0; i < IMAGE_HEIGHT - FILTER_HEIGHT + 1; ++i) {
	for (int j = 0; j < IMAGE_WIDTH - FILTER_WIDTH + 1; ++j) {
	  int sum = 0;
	  for (int fi = 0; fi < FILTER_HEIGHT; ++fi) {
		for (int fj = 0; fj < FILTER_WIDTH - FILTER_WIDTH % 4; fj += 4) {
//          sum += image[(i + fi) * IMAGE_WIDTH + j + fj] * filter[fi * FILTER_WIDTH + fj];
		  int32x4_t image_vec = vld1q_s32(&image[(i + fi) * IMAGE_WIDTH + j + fj]);
		  int32x4_t filter_vec = vld1q_s32(&filter[fi * FILTER_WIDTH + fj]);
		  int32x4_t product = vmulq_s32(image_vec, filter_vec);

		  // NEON SIMD 연산 결과를 수평으로 더합니다.
		  int32x2_t sum_lane = vpadd_s32(vget_low_s32(product), vget_high_s32(product));
		  sum += vget_lane_s32(sum_lane, 0) + vget_lane_s32(sum_lane, 1);
		}
		int fj = FILTER_WIDTH - FILTER_WIDTH % 4;
		while (fj < FILTER_WIDTH) {
		  sum += image[(i + fi) * IMAGE_WIDTH + j + fj] * filter[fi * FILTER_WIDTH + fj];
		  ++fj;
		}
	  }
	  result[i * (IMAGE_WIDTH - FILTER_WIDTH + 1) + j] = sum;
	}
  }
}

void convolution_simd_16(const std::vector<int>& image,
						 const std::vector<int>& filter,
						 std::vector<int>& result) {
  for (int i = 0; i < IMAGE_HEIGHT - FILTER_HEIGHT + 1; ++i) {
	for (int j = 0; j < IMAGE_WIDTH - FILTER_WIDTH + 1; ++j) {
	  int sum = 0;
	  for (int fi = 0; fi < FILTER_HEIGHT; ++fi) {
		for (int fj = 0; fj < FILTER_WIDTH - FILTER_WIDTH % 16; fj += 16) {
		  int32x4x4_t image_vec = vld4q_s32(&image[(i + fi) * IMAGE_WIDTH + j + fj]);
		  int32x4x4_t filter_vec = vld4q_s32(&filter[fi * FILTER_WIDTH + fj]);

		  int32x4_t product = vmulq_s32(image_vec.val[0], filter_vec.val[0]);
		  int32x2_t sum_lane = vpadd_s32(vget_low_s32(product), vget_high_s32(product));
		  sum += vget_lane_s32(sum_lane, 0) + vget_lane_s32(sum_lane, 1);

		  product = vmulq_s32(image_vec.val[1], filter_vec.val[1]);
		  sum_lane = vpadd_s32(vget_low_s32(product), vget_high_s32(product));
		  sum += vget_lane_s32(sum_lane, 0) + vget_lane_s32(sum_lane, 1);

		  product = vmulq_s32(image_vec.val[2], filter_vec.val[2]);
		  sum_lane = vpadd_s32(vget_low_s32(product), vget_high_s32(product));
		  sum += vget_lane_s32(sum_lane, 0) + vget_lane_s32(sum_lane, 1);

		  product = vmulq_s32(image_vec.val[3], filter_vec.val[3]);
		  sum_lane = vpadd_s32(vget_low_s32(product), vget_high_s32(product));
		  sum += vget_lane_s32(sum_lane, 0) + vget_lane_s32(sum_lane, 1);
		}
		int fj = FILTER_WIDTH - FILTER_WIDTH % 16;
		while (fj < FILTER_WIDTH) {
		  sum += image[(i + fi) * IMAGE_WIDTH + j + fj] * filter[fi * FILTER_WIDTH + fj];
		  ++fj;
		}
	  }
	  result[i * (IMAGE_WIDTH - FILTER_WIDTH + 1) + j] = sum;
	}
  }
}

void convolution_opencl(const std::vector<int>& image,
						const std::vector<int>& filter,
						std::vector<int>& result) {
  cl_uint num_platforms = 0;
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem image_buffer = NULL;
  cl_mem filter_buffer = NULL;
  cl_mem result_buffer = NULL;
  cl_int ret;

  // 플랫폼 가져오기
  ret = clGetPlatformIDs(1, NULL, &num_platforms);
  ret = clGetPlatformIDs(1, &platform_id, NULL);
  // 디바이스 가져오기
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
  // 컨텍스트 생성
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  // 커맨드 큐 생성
  queue = clCreateCommandQueue(context, device_id, 0, &ret);

  // 커널 소스 코드 정의
  const char* kernel_code =
	  "__kernel void convolution(__global const int* image, __global const int* filter, __global int* result) {"
	  "  int i = get_global_id(0);"
	  "  int j = get_global_id(1);"
	  "  int sum = 0;"
	  "  for (int fi = 0; fi < 3; ++fi) {"
	  "    for (int fj = 0; fj < 3; ++fj) {"
	  "      sum += image[(i + fi) * 512 + j + fj] * filter[fi * 3 + fj];"
	  "    }"
	  "  }"
	  "  result[i * (512 - 3 + 1) + j] = sum;"
	  "}";

  // 프로그램 생성
  program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &ret);
  // 프로그램 빌드
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  // 커널 생성
  kernel = clCreateKernel(program, "convolution", &ret);

  // 버퍼 생성
  image_buffer = clCreateBuffer(context,
								CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								sizeof(int) * image.size(),
								(void*)&image[0],
								&ret);
  filter_buffer = clCreateBuffer(context,
								 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								 sizeof(int) * filter.size(),
								 (void*)&filter[0],
								 &ret);
  result_buffer =
	  clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * result.size(), NULL, &ret);

  // 커널 아규먼트 설정
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_buffer);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&filter_buffer);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&result_buffer);

  // 커널 실행
  size_t global_item_size[2] = {IMAGE_WIDTH - FILTER_WIDTH + 1, IMAGE_HEIGHT - FILTER_HEIGHT + 1};
  size_t local_item_size[2] = {1, 1};
  ret = clEnqueueNDRangeKernel(queue,
							   kernel,
							   2,
							   NULL,
							   global_item_size,
							   local_item_size,
							   0,
							   NULL,
							   NULL);

  // 결과 버퍼 읽기
  ret = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
							sizeof(int) * result.size(), &result[0], 0, NULL, NULL);

  // 자원 정리
  ret = clFlush(queue);
  ret = clFinish(queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(image_buffer);
  ret = clReleaseMemObject(filter_buffer);
  ret = clReleaseMemObject(result_buffer);
  ret = clReleaseCommandQueue(queue);
  ret = clReleaseContext(context);
}
