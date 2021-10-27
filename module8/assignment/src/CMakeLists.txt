
add_executable(convolution2d convolution2d.cu)
target_link_libraries(convolution2d PRIVATE ${CUDA_CUDART_LIBRARY} ${CUDA_CUFFT_LIBRARIES})
set_property(TARGET convolution2d PROPERTY CUDA_STANDARD 11)

add_executable(matrix_scale matrix_scale.cu)
target_link_libraries(matrix_scale PRIVATE ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
set_property(TARGET matrix_scale PROPERTY CUDA_STANDARD 11)
