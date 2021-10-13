#include <iostream>
#include <cstdlib>

#include "shared_mem_cuda_kernels.h"
#include "argument_parser.h"


/*
 * Main function
 * @param argc, number of command line args
 * @param argv, 2D character array representing the commands passed via command line.
 */
int main(int argc, char** argv) {
	// Default values
	int array_size = 1024;
	int num_blocks = 1;
	std::string op = "";

	// read command line arguments
	ArgumentParser parser(argc, argv);
	if (parser.exists("-s"))
		array_size = std::atoi(parser.get_option("-s").c_str());
	if (parser.exists("-b"))
		num_blocks = std::atoi(parser.get_option("-b").c_str());
	if (parser.exists("-o"))
		op = parser.get_option("-o");
	else {
		std::cout << "[ERROR]: No operation indicated.\n";
		return EXIT_FAILURE;
	}

	// Calculate total number of threads
	int total_threads = array_size/num_blocks;
	std::cout << "total threads: " << total_threads << std::endl;
	std::cout << "Total samps proc: " << total_threads*num_blocks << std::endl;

	// Get device propoerties
	cudaDeviceProp prop; 
  int which_device; 
  cudaGetDeviceCount(&which_device); 
  cudaGetDeviceProperties(&prop, which_device);

	// Allocate memory in device for variables
  int *device_a, *device_b, *device_result; 
  cudaMalloc( ( void**)& device_a, 			array_size * sizeof ( *device_a ) ); 
  cudaMalloc( ( void**)& device_b, 			array_size * sizeof ( *device_b ) ); 
  cudaMalloc( ( void**)& device_result, array_size * sizeof ( *device_result ) ); 

	// Allocate memory on host
  int *host_a, *host_b, *host_result; 
  cudaHostAlloc((void **)&host_a, 			array_size * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_b, 			array_size * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_result, 	array_size * sizeof(int), cudaHostAllocDefault);

	// Fill input arrays
	for (int i = 0; i < array_size; i++) {
		host_a[i] = i;
		host_b[i] = rand() % (3-0+1) + 0;
	}

	// Print input variables
	std::cout << "Inputs:\n";
	for (int i = array_size - 10; i < array_size; i++) {
		std::cout << host_a[i] << "\t" << host_b[i] << std::endl;
	}

	// Create streams
  cudaStream_t stream; 
  cudaStreamCreate(&stream); 

	// Create events for timing
	cudaEvent_t start, stop, kernel_start, kernel_stop; 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);

  // Create start event
  cudaEventRecord(start);

  // Copy data to device
  cudaMemcpyAsync(device_a, host_a, array_size * sizeof ( int ), cudaMemcpyHostToDevice, stream); 
  cudaMemcpyAsync(device_b, host_b, array_size * sizeof ( int ), cudaMemcpyHostToDevice, stream); 

	// Record start of kernel time
  cudaEventRecord(kernel_start);

	// Select which kernel
	if (op == "add") {
		add_kernel_shr_i32<<< num_blocks, total_threads, 3*total_threads*sizeof(int), stream >>>(device_result, device_a, device_b, array_size);
	}
	else if (op == "sub") {
		sub_kernel_shr_i32<<< num_blocks, total_threads, 3*total_threads*sizeof(int), stream >>>(device_result, device_a, device_b, array_size);
	}
	else if (op == "mul") {
		mul_kernel_shr_i32<<< num_blocks, total_threads, 3*total_threads*sizeof(int), stream >>>(device_result, device_a, device_b, array_size);
	}
	else if (op == "mod") {
		mod_kernel_shr_i32<<< num_blocks, total_threads, 3*total_threads*sizeof(int), stream >>>(device_result, device_a, device_b, array_size);
	}

	// Record stop of kernel
	cudaEventRecord(kernel_stop);

	// Asynchronously copy data to host
  cudaMemcpyAsync(host_result, device_result, array_size * sizeof(int), cudaMemcpyDeviceToHost, stream);

	// Synchronize stream
  cudaStreamSynchronize(stream);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 

	// Calculate elapsed time
  float elapsed_time_transfer, elapsed_time_kernel = 0.0f;
  cudaEventElapsedTime(&elapsed_time_transfer, start, stop); 
  cudaEventElapsedTime(&elapsed_time_kernel, kernel_start, kernel_stop); 

	// Print result
	std::cout << "Result:\n";
	for (int i = array_size - 10; i < array_size; i++) {
		std::cout << host_result[i] << std::endl;
	}

	// Print times
	std::cout << "Time to process without transfer [" << array_size << "] samples (" << elapsed_time_kernel << ") ms\n";
	std::cout << "Time to process with transfer [" << array_size << "] samples {" << elapsed_time_transfer << "} ms\n";

	// Release memory on host
  cudaFreeHost(host_a); 
  cudaFreeHost(host_b); 
  cudaFreeHost(host_result); 

	// Release memory on device
  cudaFree(device_a); 
  cudaFree(device_b); 
  cudaFree(device_result);

	return EXIT_SUCCESS;
}
