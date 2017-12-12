cmake_minimum_required(VERSION 3.8)
project(Assignment_7)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c functions_schema.c utils.h functions_schema.h functions_database.c functions_database.h
        functions_records.c functions_records.h utils.c functions_field_list.c functions_field_list.h
        functions_linked_list.c functions_linked_list.h)
add_executable(Assignment_7 ${SOURCE_FILES})
