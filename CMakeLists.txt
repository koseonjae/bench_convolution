cmake_minimum_required(VERSION 3.24)

project(test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -O3")

enable_language(C CXX)

enable_testing()
find_package(GTest CONFIG REQUIRED)

find_package(OpenCL REQUIRED)

add_executable(test_convolutions main.cpp)

target_link_libraries(test_convolutions
    PRIVATE
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    OpenCL::OpenCL)

add_test(AllTestsInMain test_convolutions)
