# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("unittest")

set(VCPKG_ROOT "/Users/koseonjae/workspace/vcpkg" CACHE STRING "")
include(${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

option(APPLY_OPTIMIZATION "Apply compiler optimization." ON)
if (APPLY_OPTIMIZATION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

find_package(OpenCL REQUIRED)
find_package(fmt CONFIG REQUIRED)

find_package(googletest REQUIRED CONFIG)
find_package(junit-gtest REQUIRED CONFIG)

add_library(convolutions STATIC convolutions.cpp)

target_link_libraries(convolutions PRIVATE
    android
    log
    OpenCL::OpenCL
    fmt::fmt)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
    unittest

    # Sets the library as a shared library.
    SHARED

    # Provides a relative path to your source file(s).
    native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
    log-lib

    # Specifies the name of the NDK library that
    # you want CMake to locate.
    log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
    unittest

    # Links the target library to the log library
    # included in the NDK.
    ${log-lib}
    convolutions)

add_library(app_tests SHARED convolutions_test.cpp)
target_link_libraries(app_tests
    PRIVATE
    convolutions
    googletest::gtest
    junit-gtest::junit-gtest
    )
