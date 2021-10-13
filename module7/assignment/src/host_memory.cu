#include <iostream>
#include <cstdlib>
#include <chrono>

#include "cuda_kernels.h"
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
	int *in1, *in2, *res;
	std::chrono::high_resolution_clock::time_point start_w_transfer = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&in1, sizeof(int)*array_size);
	cudaMalloc((void**)&in2, sizeof(int)*array_size);
	cudaMalloc((void**)&res, sizeof(int)*array_size);
	cudaMemcpy(in1, input1_host, sizeof(int)*array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(in2, input2_host, sizeof(int)*array_size, cudaMemcpyHostToDevice);

	// Take time without transfer
	std::chrono::high_resolution_clock::time_point start_no_transfer = std::chrono::high_resolution_clock::now();

	// Select which kernel
	if (op == "add") {
		add_kernel_i32<<<num_blocks, total_threads>>>(res, in1, in2, array_size);
	}
	else if (op == "sub") {
		sub_kernel_i32<<<num_blocks, total_threads>>>(res, in1, in2, array_size);
	}
	else if (op == "mul") {
		mul_kernel_i32<<<num_blocks, total_threads>>>(res, in1, in2, array_size);
	}
	else if (op == "mod") {
		mod_kernel_i32<<<num_blocks, total_threads>>>(res, in1, in2, array_size);
	}
	
	// Take time without transfer
	std::chrono::high_resolution_clock::time_point end_no_transfer = std::chrono::high_resolution_clock::now();

	// Copy data from device to host
	cudaMemcpy(result_host, res, sizeof(int)*array_size, cudaMemcpyDeviceToHost );

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

	// Free memory
	cudaFree(in1);
	cudaFree(in2);
	cudaFree(res);

	return EXIT_SUCCESS;
}
