#include <iostream>
#include <cstdlib>
#include <chrono>


void add(int* result, const int* summand1, const int* summand2, int N) {
	for (int i = 0; i < N; i++) {
		result[i] = summand1[i] + summand2[i];
	}
}

void sub(int* result, const int* minuend, const int* subtrahend, int N) {
	for (int i = 0; i < N; i++) {
		result[i] = minuend[i] - subtrahend[i];
	}
}

void mul(int* result, const int* multiplier, const int* multiplicand, int N) {
	for (int i = 0; i < N; i++) {
		result[i] = multiplier[i] * multiplicand[i];
	}
}

void mod(int* result, const int * in, const int * modulo, int N) {
	for (int i = 0; i < N; i++) {
		if (modulo[i] == 0)
			result[i] = -1;
		else
			result[i] = in[i] % modulo[i];
	}
}

/*
 * Main function
 * @param argc, number of command line args
 * @param argv, 2D character array representing the commands passed via command line.
 */
int main(int argc, char** argv)
{
	// read command line arguments
	int array_size = 1024*1024*4;
	int op = -1;

	if (argc >= 2) {
		array_size = atoi(argv[1]);
		std::cout << "Num inputs: " << array_size << std::endl;
	}

	if (argc >= 3) {
		if (strncmp(argv[2], "add", 3) == 0) {
			std::cout << "Set to add\n";
			op = 0;
		}
		if (strncmp(argv[2], "sub", 3) == 0) {
			std::cout << "Set to sub\n";
			op = 1;
		}
		if (strncmp(argv[2], "mul", 3) == 0) {
			std::cout << "Set to mul\n";
			op = 2;
		}
		if (strncmp(argv[2], "mod", 3) == 0) {
			std::cout << "Set to mod\n";
			op = 3;
		}
	}

	// Declare pointers for GPU based params
	int *in1 = new int[array_size];
	int *in2 = new int[array_size];
	int *res = new int[array_size];

	// Fill input arrays
	for (int i = 0; i < array_size; i++) {
		in1[i] = i;
		in2[i] = rand() % (3-0+1) + 0;
	}

	std::cout << "Inputs:\n";
	for (int i = array_size - 10; i < array_size; i++) {
		std::cout << in1[i] << "\t" << in2[i] << std::endl;
	}


	// Select which kernel
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	switch(op) {
		case 0:
			add(res, in1, in2, array_size);
			break;
		case 1:
			sub(res, in1, in2, array_size);
			break;
		case 2:
			mul(res, in1, in2, array_size);
			break;
		case 3:
			mod(res, in1, in2, array_size);
			break;
		default:
			std::cout << "ERROR: No operation indicated.\n";
			return EXIT_FAILURE;
	}
	
	// Determine time
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	
	// Print result
	std::cout << "Result:\n";
	for (int i = array_size -10; i < array_size; i++) {
		std::cout << res[i] << std::endl;
	}

	// Print time
	std::cout << "Time to process [" << array_size << "] samples {" << 1000*diff.count() << "} ms\n";

	return EXIT_SUCCESS;
}
