cmake_minimum_required(VERSION 3.8)
project(Assignment_4)

set(CMAKE_C_STANDARD 99)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOURCE_FILES main.c)
add_executable(Assignment_4 ${SOURCE_FILES})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)