#include <stdlib.h>
#include <stdio.h>

#include "argument_parser.h"


/*
 * CUDA kernel for addition of a valua to an array
 * @param summand1, first array to add
 * @param summand2, constant value to add to all elements
 * @param result, the address to where store the result
 */
__global__ 
void add_const_kernel(char* result, const char* summand1, const char summand2) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = summand1[thread_idx] + summand2;
}

/*
 * CUDA kernel for subtraction of a constant value from an array
 * @param minuend, first array to subtract from
 * @param subtrahend, constant value to subtract from array
 * @param result, the address to where store the result
 */
__global__ 
void sub_const_kernel(char* result, const char* minuend, const char subtrahend) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = minuend[thread_idx] - subtrahend;
}


/*
 * Main function
 * @param argc, number of command line args
 * @param argv, 2D character array representing the commands passed via command line.
 */
int main(int argc, char** argv) {
	// Default values
	char shift_amount = 0;
	int total_threads = 1024;
	int block_size = 1;

	// read command line arguments
	ArgumentParser parser(argc, argv);
	if (parser.exists("-t"))
		total_threads = std::atoi(parser.get_option("-t").c_str());
	if (parser.exists("-b"))
		block_size = std::atoi(parser.get_option("-b").c_str());
	if (parser.exists("-s"))
		shift_amount = (char)std::atoi(parser.get_option("-s").c_str());

	int num_blocks = total_threads/block_size;

	// validate command line arguments
	if (total_threads % block_size != 0) {
		++num_blocks;
		total_threads = num_blocks*block_size;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", total_threads);
	}

	// Fill input arrays
	const char input_host[256] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
	char result_host[256];
	printf("%s\n", input_host);

	// Declare pointers for GPU based params
	char* in1, *res;

	// Copy data to device
	cudaMalloc((void**)&in1, sizeof(char)*256);
	cudaMalloc((void**)&res, sizeof(char)*256);
	cudaMemcpy(in1, input_host, sizeof(char)*256, cudaMemcpyHostToDevice);

	// Add offset for Caesar cypher
	add_const_kernel<<<num_blocks, total_threads>>>(res, in1, shift_amount);

	// Copy data from device to host
	cudaMemcpy(result_host, res, sizeof(char)*256, cudaMemcpyDeviceToHost);
	
	// Print result
	printf("Encoded: %s\n", result_host);

	// Subtract Caesar cypher
	sub_const_kernel<<<num_blocks, total_threads>>>(in1, res, shift_amount);

	// Copy results from device to host
	cudaMemcpy(result_host, in1, sizeof(char)*256, cudaMemcpyDeviceToHost);

	// Print result
	printf("Decoded: %s\n", result_host);

	// Free memory
	cudaFree(in1);
	cudaFree(res);

	return EXIT_SUCCESS;
}
