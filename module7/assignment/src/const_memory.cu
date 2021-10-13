#include <iostream>
#include <cstdlib>
#include <chrono>

#include "argument_parser.h"
#include "cuda_kernels.h"


__constant__ int gpu_const_mem[16384];

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

	// Calculate number of threads
	int total_threads = 1 + ((array_size-1)/num_blocks);
	std::cout << "total threads: " << total_threads << std::endl;

	// Declare pointers for GPU based params
	int *input1_host = new int[array_size];
	int *input2_host = new int[array_size];
	int *result_host = new int[array_size];

	// Fill input arrays
	for (int i = 0; i < array_size; i++) {
		input1_host[i] = i;
		input2_host[i] = rand() % (3-0+1) + 0;
	}
	std::cout << "Inputs:\n";
	for (int i = array_size - 10; i < array_size; i++) {
		std::cout << input1_host[i] << "\t" << input2_host[i] << std::endl;
	}

	// Copy data to device
	int *cont_mem_ptr;
	std::chrono::high_resolution_clock::time_point start_w_transfer = std::chrono::high_resolution_clock::now();
	cudaGetSymbolAddress((void **)&cont_mem_ptr, gpu_const_mem);
	cudaMemcpy(cont_mem_ptr, input1_host, array_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&cont_mem_ptr[array_size], input2_host, array_size*sizeof(int), cudaMemcpyHostToDevice);

	// Take time without transfer
	std::chrono::high_resolution_clock::time_point start_no_transfer = std::chrono::high_resolution_clock::now();

	// Select which kernel
	if (op == "add") {
		add_kernel_i32<<<num_blocks, total_threads>>>(cont_mem_ptr, cont_mem_ptr, &cont_mem_ptr[array_size], array_size);
	}
	else if (op == "sub") {
		sub_kernel_i32<<<num_blocks, total_threads>>>(cont_mem_ptr, cont_mem_ptr, &cont_mem_ptr[array_size], array_size);
	}
	else if (op == "mul") {
		mul_kernel_i32<<<num_blocks, total_threads>>>(cont_mem_ptr, cont_mem_ptr, &cont_mem_ptr[array_size], array_size);
	}
	else if (op == "mod") {
		mod_kernel_i32<<<num_blocks, total_threads>>>(cont_mem_ptr, cont_mem_ptr, &cont_mem_ptr[array_size], array_size);
	}
	
	// Take time without transfer
	std::chrono::high_resolution_clock::time_point end_no_transfer = std::chrono::high_resolution_clock::now();

	// Copy data from device to host
	cudaMemcpy(result_host, cont_mem_ptr, sizeof(int)*array_size, cudaMemcpyDeviceToHost);

	// Determine time
	std::chrono::high_resolution_clock::time_point end_w_transfer = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_no_transfer = end_no_transfer - start_no_transfer;
	std::chrono::duration<double> diff_w_transfer = end_w_transfer - start_w_transfer;
	
	// Print result
	std::cout << "Result:\n";
	for (int i = array_size -10; i < array_size; i++) {
		std::cout << result_host[i] << std::endl;
	}

	// Print time
	std::cout << "Time to process without transfer [" << array_size << "] samples (" << 1000*diff_no_transfer.count() << ") ms\n";
	std::cout << "Time to process with transfer [" << array_size << "] samples {" << 1000*diff_w_transfer.count() << "} ms\n";

	cudaDeviceReset();

	return EXIT_SUCCESS;
}
