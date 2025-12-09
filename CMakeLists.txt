cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dlrm-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# LibTorch 경로 (libtorch 디렉토리)
set(CMAKE_PREFIX_PATH "/home/southbell/libtorch")

find_package(Torch REQUIRED)  

add_executable(dlrm-cuda dlrm-cuda.cu)

# Torch 라이브러리 링크
target_link_libraries(dlrm-cuda "${TORCH_LIBRARIES}")

# nvcc 에도 Torch include 경로가 전달되도록
target_include_directories(dlrm-cuda PRIVATE "${TORCH_INCLUDE_DIRS}")

# CUDA 분리 컴파일 옵션 (복잡한 프로젝트에서 유용)
set_property(TARGET dlrm-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)