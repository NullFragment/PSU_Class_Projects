cmake_minimum_required(VERSION 3.8)
project(Assignment_5)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(SOURCE_FILES prefix_sum.cu)
add_executable(Assignment_5 ${SOURCE_FILES})