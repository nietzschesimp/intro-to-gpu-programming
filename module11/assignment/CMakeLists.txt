
cmake_minimum_required(VERSION 3.0)
project(opecl-module10 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 11)

# Solution from stackoverflow question
# https://stackoverflow.com/questions/67641589/
#    how-can-i-tell-cmake-3-9-i-want-to-use-nvidias-opencl-library
find_package(CUDA 8.0 REQUIRED)

find_path(
  OpenCL_INCLUDE_DIR CL/opencl.h
  HINTS "${CUDA_TOOLKIT_ROOT_DIR}/include"
  NO_DEFAULT_PATH
)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(opencl_lib64 "${CUDA_TOOLKIT_ROOT_DIR}/lib64" )
endif()

find_library(
  OpenCL_LIBRARY OpenCL
  HINTS ${opencl_lib64} "${CUDA_TOOLKIT_ROOT_DIR}/lib"
  PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
  NO_DEFAULT_PATH
)

find_package(OpenCL REQUIRED)

include_directories(include)
add_subdirectory(src)
