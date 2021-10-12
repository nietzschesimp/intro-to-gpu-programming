#ifndef REGISTER_CUDA_KERNELS_H_
#define REGISTER_CUDA_KERNELS_H_

/*
 * CUDA kernel for addition of two arrays of 32 bit integes
 * @param summand1, first array to add
 * @param summand2, second array to add
 * @param result, the address to where store the result
 */
__global__ 
void add_kernel_register_i32(int* result, const int* summand1, const int* summand2, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		int tmp_summand1 = summand1[thread_idx];
		int tmp_summand2 = summand2[thread_idx];
		tmp_summand1 = tmp_summand1 + tmp_summand2;
		result[thread_idx] = tmp_summand1;
	}
}

/*
 * CUDA kernel for subtraction of two arrays of 32 bit integes
 * @param minuend, first array to subtract from
 * @param subtrahend, second array to subtract from first
 * @param result, the address to where store the result
 */
__global__ 
void sub_kernel_register_i32(int* result, const int* minuend, const int* subtrahend, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		int tmp_minuend = minuend[thread_idx];
		int tmp_subtrahend = subtrahend[thread_idx];
		tmp_minuend = tmp_minuend - tmp_subtrahend;
		result[thread_idx] = tmp_minuend;
	}
}

/*
 * CUDA kernel for multiplication of two arrays of 32 bit integes
 * @param multiplier, first array to multiply
 * @param summand2, second array of values to multiply with
 * @param result, the address to where store the result
 */
__global__ 
void mul_kernel_register_i32(int* result, const int* multiplier, const int* multiplicand, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		int tmp_multiplier = multiplier[thread_idx];
		int tmp_multiplicand = multiplicand[thread_idx];
		tmp_multiplier = tmp_multiplier * tmp_multiplicand;
		result[thread_idx] = tmp_multiplier;
	}
}

/*
 * CUDA kernel for modulo of two arrays of 32 bit integes
 * @param in, first array to take modulo from
 * @param modulo, second array of values corresponding to modulo bases
 * @param result, the address to where store the result
 */
__global__
void mod_kernel_register_i32(int* result, const int* in, const int* modulo, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thread_idx < size) {
		int tmp_in = in[thread_idx];
		int tmp_modulo = modulo[thread_idx];
		tmp_in = tmp_in % tmp_modulo;
		result[thread_idx] = tmp_in;
	}
}

#endif
