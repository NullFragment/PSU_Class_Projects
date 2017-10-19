cmake_minimum_required(VERSION 3.8)
project(Assignment_5)

set(CMAKE_C_STANDARD 11)

set(SOURCE_FILES main.c functions_schema.c utils.h functions_schema.h functions_database.c functions_database.h functions_records.c functions_records.h utils.c)
add_executable(Assignment_5 ${SOURCE_FILES})