#include "adder.h"

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

constexpr int IMAGE_HEIGHT = 1024;
constexpr int IMAGE_WIDTH = 1024;
constexpr int FILTER_HEIGHT = 19;
constexpr int FILTER_WIDTH = 19;

#include <fmt/format.h>

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