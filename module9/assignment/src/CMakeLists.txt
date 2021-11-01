
add_executable(operations_thrust vector_operations_thrust.cu)
target_link_libraries(operations_thrust PUBLIC ${CUDA_CUDART_LIBRARY})
target_include_directories(operations_thrust PUBLIC ${CUDA_INCLUDE_DIRS})
set_property(TARGET operations_thrust PROPERTY CUDA_STANDARD 11)

add_executable(npp_convolution npp_convolution.cu)
target_link_libraries(npp_convolution PUBLIC ${CUDA_CUDART_LIBRARY} 
	${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppif_LIBRARY}
	${CMAKE_SOURCE_DIR}/extern/FreeImage/lib/linux/x86_64/libfreeimage.a)
target_include_directories(npp_convolution PUBLIC ${CUDA_INCLUDE_DIRS})
set_property(TARGET npp_convolution PROPERTY CUDA_STANDARD 11)

#add_executable(triangle_count triangle_count.cu)
#target_link_libraries(triangle_count PUBLIC ${CUDA_CUDART_LIBRARY} nvgraph)
#target_include_directories(triangle_count PUBLIC ${CUDA_INCLUDE_DIRS})
#set_property(TARGET triangle_count PROPERTY CUDA_STANDARD 11)
