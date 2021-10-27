#include <iostream>
#include <cublas.h>
#include <string>

#include "argument_parser.h"


int main(int argc, char** argv) {
	// Default values
	int no_rows = 1024;
	int no_cols = 1024;
	float scalar = 3.0;

	// read command line arguments
	ArgumentParser parser(argc, argv);
	if (parser.exists("-r"))
		no_rows = std::stoi(parser.get_option("-r"));
	if (parser.exists("-c"))
		no_cols = std::stoi(parser.get_option("-c"));
	if (parser.exists("-s"))
		scalar = std::stof(parser.get_option("-s"));
	
  // Create memory for input data
	float* host_input;
	float* device_input;
  cudaMallocHost((void**)&host_input, sizeof(float)*no_rows*no_cols);
  cudaHostGetDevicePointer(&device_input, host_input, 0);

	// Populate input array
	for(int ii = 0; ii < no_rows; ii++) {
		for(int jj = 0; jj < no_cols; jj++) {
			*(host_input + no_rows*ii + jj) = (float)(rand() % 2);
		}
	}

	std::cout << "INPUT:\n";
	for(int ii=0; ii < 10; ii++) {
		for(int jj=0; jj < 10; jj++) {
			std::cout << *(host_input + no_rows*ii + jj) << ", ";
		}
		std::cout << std::endl;
	}

	cublasStatus status = cublasInit();	
	if (status != CUBLAS_STATUS_SUCCESS) { 
		return -1; 
	}

	// Create events for timing
	cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Create start event
  cudaEventRecord(start);

	cublasSscal(no_rows*no_cols, scalar, device_input, 1);

	cudaEventRecord(stop);

	cublasShutdown();

	std::cout << "OUTPUT:\n";
	for(int ii=0; ii < 10; ii++) {
		for(int jj=0; jj < 10; jj++) {
			std::cout << *(host_input + no_rows*ii + jj) << ", ";
		}
		std::cout << std::endl;
	}

	float elapsed_time = 0.0f;
  cudaEventElapsedTime(&elapsed_time, start, stop); 
	std::cout << "Time to process without transfer [" << no_rows*no_cols << "] samples (" << elapsed_time << ") ms\n";

	return 0;
}
