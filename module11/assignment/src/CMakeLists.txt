
# Create executable for vector operations
add_executable(convolution Convolution.cpp)
target_include_directories(convolution PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(convolution PUBLIC ${OpenCL_LIBRARIES})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Convolution.cl ${CMAKE_BINARY_DIR}/src)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Scaling.cl ${CMAKE_BINARY_DIR}/src)

