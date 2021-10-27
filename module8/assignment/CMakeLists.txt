cmake_minimum_required(VERSION 3.0)
project (mod8 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(include ${CUDA_INCLUDE_DIRS})

add_subdirectory(src)

