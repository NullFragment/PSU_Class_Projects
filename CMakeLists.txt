cmake_minimum_required(VERSION 3.8)
project(CUDA_Neural_Network)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(SOURCE_FILES src/cuda_nn.cu)
add_executable(CUDA_Neural_Network ${SOURCE_FILES})