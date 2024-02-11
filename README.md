# bench_convolution
This repository compares each convolution implementation's performance such as 
- cpu scalar
- cpu simd with 4 elements fetching
- cpu simd with 16 elements fethcing
- opencl

### env
- Apple M1 Pro

### Dependencies using vcpkg
- gtest
- opencl

### Result
Debug
```
convolution_cpu took 1272837 microseconds.
convolution_cpu_cache took 1346446 microseconds.
convolution_cpu_cache_optimized took 1306749 microseconds.
convolution_simd_4 took 1839025 microseconds.
convolution_simd_16 took 1359168 microseconds.
convolution_opencl took 534 microseconds.
```
Release
```
convolution_cpu took 27135 microseconds.
convolution_cpu_cache took 27625 microseconds.
convolution_cpu_cache_optimized took 27060 microseconds.
convolution_simd_4 took 35290 microseconds.
convolution_simd_16 took 55098 microseconds.
convolution_opencl took 34 microseconds.
```
