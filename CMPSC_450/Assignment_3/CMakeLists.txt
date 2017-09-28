cmake_minimum_required(VERSION 3.8)
project(Assignment_3)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(Assignment_3 ${SOURCE_FILES})