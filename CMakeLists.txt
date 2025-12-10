cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dlrm-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# LibTorch 경로 (libtorch 디렉토리)
set(CMAKE_PREFIX_PATH "/home/southbell/libtorch")

find_package(Torch REQUIRED)  

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(dlrm-cuda dlrm-cuda.cu)

# Torch 라이브러리 링크
target_link_libraries(dlrm-cuda "${TORCH_LIBRARIES}")

target_include_directories(dlrm-cuda PRIVATE "${TORCH_INCLUDE_DIRS}")

set_property(TARGET dlrm-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)