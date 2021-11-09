cmake_minimum_required(VERSION 3.0)
project(opecl-module10 LANGUAGES CXX)

find_package(OpenCL REQUIRED)

include_directories(include)
add_subdirectory(src)
