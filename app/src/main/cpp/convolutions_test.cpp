#include "convolutions.h"

#include "gtest/gtest.h"

class ConvolutionTest : public ::testing::Test {
 protected:
  void
  SetUp() override {
	image = generateRandomImage();
	filter = std::vector<int>(FILTER_HEIGHT * FILTER_WIDTH, 1);
	result =
		std::vector<int>((IMAGE_HEIGHT - FILTER_HEIGHT + 1) * (IMAGE_WIDTH - FILTER_WIDTH + 1));
  }

 public:
  std::vector<int> image;
  std::vector<int> filter;
  std::vector<int> result;
};

TEST_F(ConvolutionTest, convolution_cpu) {
  auto timer = Timer("convolution_cpu");
  convolution_cpu(image, filter, result);
}

TEST_F(ConvolutionTest, convolution_cpu_cache) {
  auto timer = Timer("convolution_cpu_cache");
  convolution_cpu_cache(image, filter, result);
}

TEST_F(ConvolutionTest, convolution_cpu_cache_optimized) {
  auto timer = Timer("convolution_cpu_cache_optimized");
  convolution_cpu_cache_optimized(image, filter, result);
}

TEST_F(ConvolutionTest, convolution_simd_4) {
  auto timer = Timer("convolution_simd_4");
  convolution_simd_4(image, filter, result);
}

TEST_F(ConvolutionTest, convolution_simd_16) {
  auto timer = Timer("convolution_simd_16");
  convolution_simd_16(image, filter, result);
}

TEST_F(ConvolutionTest, convolution_opencl) {
  auto timer = Timer("convolution_opencl");
  convolution_opencl(image, filter, result);
}

TEST_F(ConvolutionTest, ResultsMatch) {
  convolution_cpu(image, filter, result);
  std::vector<int> expected_result = result;

  convolution_cpu_cache(image, filter, result);
  ASSERT_EQ(result, expected_result);

  convolution_cpu_cache_optimized(image, filter, result);
  ASSERT_EQ(result, expected_result);

  convolution_simd_4(image, filter, result);
  ASSERT_EQ(result, expected_result);

  convolution_simd_16(image, filter, result);
  ASSERT_EQ(result, expected_result);

  convolution_opencl(image, filter, result);
  ASSERT_EQ(result, expected_result);
}
