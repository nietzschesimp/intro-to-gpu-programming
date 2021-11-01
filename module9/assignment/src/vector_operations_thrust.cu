#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <iostream>
#include <string>
#include <chrono>

#include "argument_parser.h"

#define VAL_MAX 3
#define VAL_MIN 0


int randrange() {
	return VAL_MIN + (std::rand() % (VAL_MAX - VAL_MIN + 1));
}

int main(int argc, char** argv) {
	int no_samples = 1024;
	std::string op = "";
	
	// read command line arguments
	ArgumentParser parser(argc, argv);
	if (parser.exists("-s"))
		no_samples = std::stoi(parser.get_option("-s").c_str());
	if (parser.exists("-o"))
		op = parser.get_option("-o");
	else {
		std::cout << "[ERROR]: No operation indicated.\n";
		return EXIT_FAILURE;
	}
	
	thrust::host_vector<int> host_vector(no_samples);
	thrust::device_vector<int> result(no_samples);

	thrust::sequence(host_vector.begin(), host_vector.end());
	thrust::device_vector<int> sequence = host_vector;
	
	thrust::generate(host_vector.begin(), host_vector.end(), randrange);
	thrust::device_vector<int> random_input = host_vector;

	std::cout << "Inputs:\n";
	for (int i = no_samples - 10; i < no_samples; i++) {
		std::cout << i << "\t" << host_vector[i] << std::endl;
	}

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	// Select which kernel
	if (op == "add") {
		thrust::transform(random_input.begin(), random_input.end(), sequence.begin(), result.begin(), thrust::plus<int>());
	}
	else if (op == "sub") {
		thrust::transform(random_input.begin(), random_input.end(), result.begin(), thrust::negate<int>());
		thrust::transform(sequence.begin(), sequence.end(), result.begin(), result.begin(), thrust::plus<int>());
	}
	else if (op == "mul") {
		thrust::transform(random_input.begin(), random_input.end(), sequence.begin(), result.begin(), thrust::multiplies<int>());
	}
	else if (op == "mod") {
		thrust::transform(random_input.begin(), random_input.end(), sequence.begin(), result.begin(), thrust::modulus<int>());
	}
	
	// Take time without transfer
	std::chrono::high_resolution_clock::time_point end_no_transfer = std::chrono::high_resolution_clock::now();

	// Synchonize data between device and host
	thrust::copy(result.begin(), result.end(), host_vector.begin());

	// Determine time
	std::chrono::high_resolution_clock::time_point end_w_transfer = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff_no_transfer = end_no_transfer - start;
	std::chrono::duration<double> diff_w_transfer = end_w_transfer - start;
	
	// Print result
	std::cout << "Result:\n";
	for (int i = no_samples - 10; i < no_samples; i++) {
		std::cout << host_vector[i] << std::endl;
	}

	// Print times
	std::cout << "Time to process without transfer [" << no_samples << "] samples (" << 1000*diff_no_transfer.count() << ") ms\n";
	std::cout << "Time to process with transfer [" << no_samples << "] samples {" << 1000*diff_w_transfer.count() << "} ms\n";

	return 0;
}
