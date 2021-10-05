#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_

/*
 * CUDA kernel for addition of two arrays of 32 bit integes
 * @param summand1, first array to add
 * @param summand2, second array to add
 * @param result, the address to where store the result
 */
__global__ 
void add_kernel_i32(int* result, const int* summand1, const int* summand2, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		result[thread_idx] = summand1[thread_idx] + summand2[thread_idx];
	}
}

/*
 * CUDA kernel for subtraction of two arrays of 32 bit integes
 * @param minuend, first array to subtract from
 * @param subtrahend, second array to subtract from first
 * @param result, the address to where store the result
 */
__global__ 
void sub_kernel_i32(int* result, const int* minuend, const int* subtrahend, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		result[thread_idx] = minuend[thread_idx] - subtrahend[thread_idx];
	}
}

/*
 * CUDA kernel for multiplication of two arrays of 32 bit integes
 * @param multiplier, first array to multiply
 * @param summand2, second array of values to multiply with
 * @param result, the address to where store the result
 */
__global__ 
void mul_kernel_i32(int* result, const int* multiplier, const int* multiplicand, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		result[thread_idx] = multiplier[thread_idx] * multiplicand[thread_idx];
	}
}

/*
 * CUDA kernel for modulo of two arrays of 32 bit integes
 * @param in, first array to take modulo from
 * @param modulo, second array of values corresponding to modulo bases
 * @param result, the address to where store the result
 */
__global__
void mod_kernel_i32(int* result, const int* in, const int* modulo, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		result[thread_idx] = in[thread_idx] % modulo[thread_idx];
	}
}

/*
 * CUDA kernel for addition of two arrays of chars
 * @param summand1, first array to add
 * @param summand2, second array to add
 * @param result, the address to where store the result
 */
__global__ 
void add_kernel_char(char* result, const char* summand1, const char* summand2) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = summand1[thread_idx] + summand2[thread_idx];
}

/*
 * CUDA kernel for subtraction of two arrays of 32 bit integes
 * @param minuend, first array to subtract from
 * @param subtrahend, second array to subtract from first
 * @param result, the address to where store the result
 */
__global__ 
void sub_kernel_char(char* result, const char* minuend, const char* subtrahend) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = minuend[thread_idx] - subtrahend[thread_idx];
}

/*
 * CUDA kernel for multiplication of two arrays of 32 bit integes
 * @param multiplier, first array to multiply
 * @param summand2, second array of values to multiply with
 * @param result, the address to where store the result
 */
__global__ 
void mul_kernel_char(char* result, const char* multiplier, const char* multiplicand) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = multiplier[thread_idx] * multiplicand[thread_idx];
}

/*
 * CUDA kernel for modulo of two arrays of 32 bit integes
 * @param in, first array to take modulo from
 * @param modulo, second array of values corresponding to modulo bases
 * @param result, the address to where store the result
 */
__global__
void mod_kernel_char(char* result, const char* in, const char* modulo) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = in[thread_idx] % modulo[thread_idx];
}

#endif
