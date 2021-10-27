#include "argument_parser.h"
#include <iostream>
#include <cufft.h>
#include <string>


typedef float2 fc32;


float kernel[9] = {
	1, 1, 1,
	1, 0, 1,
	1, 1, 1
};

__global__ void kernel_multiply_complex_fc32(fc32* out, const fc32* in, const fc32* taps, const unsigned int size) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x ;
    if (i < size) {
      out[i].x = in[i].x * taps[i].x - in[i].y*taps[i].y;
      out[i].y = in[i].x * taps[i].y + in[i].y*taps[i].x;
    }
}

int main(int argc, char** argv) {
	// Default values
	int no_rows = 1024;
	int no_cols = 1024;

	// read command line arguments
	ArgumentParser parser(argc, argv);
	if (parser.exists("-r"))
		no_rows = std::stoi(parser.get_option("-r"));
	if (parser.exists("-c"))
		no_cols = std::stoi(parser.get_option("-c"));

  // Instantiate FFT plan
	cufftHandle fft_plan, ifft_plan;

	// Input pointers
	float* host_input;
	float* device_input;

	// Output pointers
	float* host_output;
	float* device_output;

	// Kernel taps
	fc32* device_taps;
	fc32* device_transformed;
  
	// Create fft plan
  cufftPlan2d(&fft_plan, no_rows, no_cols, CUFFT_R2C);
  cufftPlan2d(&ifft_plan, no_rows, no_cols, CUFFT_C2R);
  
  // Allocate data for kernel taps in memory
  cudaMalloc((void**)&device_taps, sizeof(fc32)*no_rows*no_cols);
  
  // Allocate data for transformed input in memory
  cudaMalloc((void**)&device_transformed, sizeof(fc32)*no_rows*no_cols);

  // Create memory for input data
  cudaMallocHost((void**)&host_input, sizeof(float)*no_rows*no_cols);
  cudaHostGetDevicePointer(&device_input, host_input, 0);

  // Create memory pointers for output
  cudaMallocHost((void**)&host_output, sizeof(float)*no_rows*no_cols);
  cudaHostGetDevicePointer(&device_output, host_output, 0);
	
	// Populate input array
	for(int ii = 0; ii < no_rows; ii++) {
		for(int jj = 0; jj < no_cols; jj++) {
			*(host_input + ii*no_rows + jj) = 0.0f;
		}
	}
	
	for(int ii=0; ii < 3; ii++) {
		for(int jj=0; jj < 3; jj++) {
			*(host_input + ii*no_rows + jj) = *(kernel + ii*3 + jj)/no_rows/no_cols;
		}
	}

	std::cout << "KERNEL:\n";
	for(int ii=0; ii < 10; ii++) {
		for(int jj=0; jj < 10; jj++) {
			std::cout << *(host_input + no_rows*ii + jj) << ", ";
		}
		std::cout << std::endl;
	}
	cudaMemcpy(device_input, host_input, no_rows*no_cols*sizeof(float), cudaMemcpyHostToDevice);

  // Transform taps
  cufftExecR2C(fft_plan, device_input, device_taps);

	// Populate input array
	for(int ii = 0; ii < no_rows; ii++) {
		for(int jj = 0; jj < no_cols; jj++) {
			*(host_input + no_rows*ii + jj) = (float)(rand() % 2);
		}
	}
	cudaMemcpy(device_input, host_input, no_rows*no_cols*sizeof(float), cudaMemcpyHostToDevice);

	// Create events for timing
	cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Create start event
  cudaEventRecord(start);

	// Transform input
  cufftExecR2C(fft_plan, device_input, device_transformed);

	// Perform complex multiplication
	int no_blocks = std::max(no_rows*no_cols/1024, 1);
	int no_threads = 1024;
	kernel_multiply_complex_fc32<<<no_threads, no_blocks>>>(device_transformed, device_transformed, device_taps, 16*no_rows*no_cols);

	// Inverse transform putput
	cufftExecC2R(ifft_plan, device_transformed, device_output);

  // Create start event
  cudaEventRecord(stop);

	// Wait for all operations in the GPU to finish
	cudaMemcpy(host_output, device_output, no_rows*no_cols*sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "INPUT:\n";
	for(int ii=0; ii < 10; ii++) {
		for(int jj=0; jj < 10; jj++) {
			std::cout << *(host_input + no_rows*ii + jj) << ", ";
		}
		std::cout << std::endl;
	}

	std::cout << "OUTPUT:\n";
	for(int ii=0; ii < 10; ii++) {
		for(int jj=0; jj < 10; jj++) {
			std::cout << *(host_output + no_rows*ii + jj) << ", ";
		}
		std::cout << std::endl;
	}

	float elapsed_time = 0.0f;
  cudaEventElapsedTime(&elapsed_time, start, stop); 
	std::cout << "Time to process without transfer [" << no_rows*no_cols << "] samples (" << elapsed_time << ") ms\n";

  cudaFree(device_transformed);
  cudaFree(device_taps);
  cudaFreeHost(host_output);
  cudaFreeHost(host_input);
	cufftDestroy(fft_plan);

	return 0;
}
