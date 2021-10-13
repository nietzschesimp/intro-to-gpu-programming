#ifndef SHARED_MEM_CUDA_KERNELS_H_
#define SHARED_MEM_CUDA_KERNELS_H_

/*
 * CUDA kernel for addition of two arrays of 32 bit integes
 * @param summand1, first array to add
 * @param summand2, second array to add
 * @param result, the address to where store the result
 */
__global__ 
void add_kernel_shr_i32(int* result, const int* summand1, const int* summand2, int size) {
	const unsigned int array_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shr[];
	if (array_idx < size) {
		shr[threadIdx.x] = summand1[array_idx];
		shr[threadIdx.x + blockDim.x] = summand2[array_idx];
		__syncthreads();
		shr[threadIdx.x + 2*blockDim.x] = shr[threadIdx.x] + shr[threadIdx.x + blockDim.x];
		__syncthreads();
		result[array_idx] = shr[threadIdx.x + 2*blockDim.x];
	}
}

/*
 * CUDA kernel for subtraction of two arrays of 32 bit integes
 * @param minuend, first array to subtract from
 * @param subtrahend, second array to subtract from first
 * @param result, the address to where store the result
 */
__global__ 
void sub_kernel_shr_i32(int* result, const int* minuend, const int* subtrahend, int size) {
	const unsigned int array_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shr[];
	if (array_idx < size) {
		shr[threadIdx.x] = minuend[array_idx];
		shr[threadIdx.x + blockDim.x] = subtrahend[array_idx];
		__syncthreads();
		shr[threadIdx.x + 2*blockDim.x] = shr[threadIdx.x] - shr[threadIdx.x + blockDim.x];
		__syncthreads();
		result[array_idx] = shr[threadIdx.x + 2*blockDim.x];
	}
}

/*
 * CUDA kernel for multiplication of two arrays of 32 bit integes
 * @param multiplier, first array to multiply
 * @param summand2, second array of values to multiply with
 * @param result, the address to where store the result
 */
__global__ 
void mul_kernel_shr_i32(int* result, const int* multiplier, const int* multiplicand, int size) {
	const unsigned int array_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shr[];
	if (array_idx < size) {
		shr[threadIdx.x] = multiplier[array_idx];
		shr[threadIdx.x + blockDim.x] = multiplicand[array_idx];
		__syncthreads();
		shr[threadIdx.x + 2*blockDim.x] = shr[threadIdx.x] * shr[threadIdx.x + blockDim.x];
		__syncthreads();
		result[array_idx] = shr[threadIdx.x + 2*blockDim.x];
	}
}

/*
 * CUDA kernel for modulo of two arrays of 32 bit integes
 * @param in, first array to take modulo from
 * @param modulo, second array of values corresponding to modulo bases
 * @param result, the address to where store the result
 */
__global__
void mod_kernel_shr_i32(int* result, const int* in, const int* modulo, int size) {
	const unsigned int array_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shr[];
	if (array_idx < size) {
		shr[threadIdx.x] = in[array_idx];
		shr[threadIdx.x + blockDim.x] = modulo[array_idx];
		__syncthreads();
		shr[threadIdx.x + 2*blockDim.x] = shr[threadIdx.x] % shr[threadIdx.x + blockDim.x];
		__syncthreads();
		result[array_idx] = shr[threadIdx.x + 2*blockDim.x];
	}
}

#endif
