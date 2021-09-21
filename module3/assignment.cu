#include <stdlib.h>
#include <stdio.h>


/*
 * CUDA kernel for addition
 * @param summand1, first array to add
 * @param summand2, second array to add
 * @param result, the address to where store the result
 */
__global__ 
void add_kernel(int* result, const int* summand1, const int* summand2) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = summand1[thread_idx] + summand2[thread_idx];
}

/*
 * CUDA kernel for subtraction
 * @param minuend, first array to subtract from
 * @param subtrahend, second array to subtract from first
 * @param result, the address to where store the result
 */
__global__ 
void sub_kernel(int* result, const int* minuend, const int* subtrahend) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = minuend[thread_idx] - subtrahend[thread_idx];
}

/*
 * CUDA kernel for multiplication
 * @param multiplier, first array to multiply
 * @param summand2, second array of values to multiply with
 * @param result, the address to where store the result
 */
__global__ 
void mul_kernel(int* result, const int* multiplier, const int* multiplicand) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = multiplier[thread_idx] * multiplicand[thread_idx];
}

/*
 * CUDA kernel for modulo
 * @param in, first array to take modulo from
 * @param modulo, second array of values corresponding to modulo bases
 * @param result, the address to where store the result
 */
__global__
void mod_kernel(int* result, const int * in, const int * modulo) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = in[thread_idx] % modulo[thread_idx];
}

/*
 * Arrays to store data for simulation
 */
#define ARRAY_SIZE 1024*1024*16
int input1_host[ARRAY_SIZE];
int input2_host[ARRAY_SIZE];
int result_host[ARRAY_SIZE];

/*
 * Main function
 * @param argc, number of command line args
 * @param argv, 2D character array representing the commands passed via command line.
 */
int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	int op = -1;

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}

	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	if (argc >= 4) {
		if (strncmp(argv[3], "add", 3) == 0) {
			printf("Set to add\n");
			op = 0;
		}
		if (strncmp(argv[3], "sub", 3) == 0) {
			printf("Set to sub\n");
			op = 1;
		}
		if (strncmp(argv[3], "mul", 3) == 0) {
			printf("Set to mul\n");
			op = 2;
		}
		if (strncmp(argv[3], "mod", 3) == 0) {
			printf("Set to mod\n");
			op = 3;
		}
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Fill input arrays
	for (int i = 0; i < ARRAY_SIZE; i++) {
		input1_host[i] = i;
		input2_host[i] = rand() % (3-0+1) + 0;
	}

	printf("Inputs:\n");
	for (int i = 0; i < 10; i++) {
		printf("%i\t%i\n", input1_host[i], input2_host[i]);
	}

	// Declare pointers for GPU based params
	int* in1;
	int* in2;
	int* res;

	// Copy data to device
	cudaMalloc((void**)&in1, sizeof(int)*ARRAY_SIZE);
	cudaMalloc((void**)&in2, sizeof(int)*ARRAY_SIZE);
	cudaMalloc((void**)&res, sizeof(int)*ARRAY_SIZE);
	cudaMemcpy(in1, input1_host, sizeof(int)*ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(in2, input2_host, sizeof(int)*ARRAY_SIZE, cudaMemcpyHostToDevice);

	// Select which kernel
	switch(op) {
		case 0:
			printf("Adding arrays\n");
			add_kernel<<<numBlocks, totalThreads>>>(res, in1, in2);
			break;
		case 1:
			printf("Subtracting arrays\n");
			sub_kernel<<<numBlocks, totalThreads>>>(res, in1, in2);
			break;
		case 2:
			printf("Multiplication arrays\n");
			mul_kernel<<<numBlocks, totalThreads>>>(res, in1, in2);
			break;
		case 3:
			printf("Modulo arrays\n");
			mod_kernel<<<numBlocks, totalThreads>>>(res, in1, in2);
			break;
		default:
			printf("ERROR: No operation indicated.\n");
			return EXIT_FAILURE;
	}
	
	// Copy data from device to host
	cudaMemcpy(result_host, res, sizeof(int)*ARRAY_SIZE, cudaMemcpyDeviceToHost );
	
	/*
	// Print result
	printf("Result:\n");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%i\n", result_host[i]);
	}
	*/

	// Free memory
	cudaFree(in1);
	cudaFree(in2);
	cudaFree(res);

	return EXIT_SUCCESS;
}
