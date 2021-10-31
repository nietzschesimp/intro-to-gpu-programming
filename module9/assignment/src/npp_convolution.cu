#include <UtilNPP/ImagesCPU.h>
#include <UtilNPP/ImagesNPP.h>
#include <UtilNPP/ImageIO.h>
#include <iostream>
#include <chrono>
#include <npp.h>

#include "argument_parser.h"


int main(int argc, char** argv) {
	std::string output_filename = "filtered_image";
	std::string image_filename;
	
	// read command line arguments
	ArgumentParser parser(argc, argv);
	if (parser.exists("-i")) {
		image_filename = parser.get_option("-i");
	}
	else {
		std::cout << "[ERROR]: No image filename indicated.\n";
		return EXIT_FAILURE;
	}

	// Find extension
	std::string::size_type dot = image_filename.rfind('.');
	std::string extension = "";
	if (dot != std::string::npos) {
    extension = image_filename.substr(dot, image_filename.size());
  }
	output_filename += extension;

	npp::ImageCPU_8u_C1 image_host;
	npp::loadImage(image_filename, image_host);
	npp::ImageNPP_8u_C1 image_device(image_host);
	NppiSize kernel_size = {3, 3};
	NppiSize roi = {(int)image_host.width() - kernel_size.width + 1, (int)image_host.height() - kernel_size.height + 1};
	npp::ImageNPP_8u_C1 filtered_device(roi.width, roi.height);
	npp::ImageCPU_8u_C1 filtered_image(filtered_device.size());

	// Allocate kernel
	Npp32s kernel_host[9] = {
		-1, -1, -1,
		-1,  8, -1,
		-1, -1, -1
	};
	Npp32s* kernel_device;
	cudaMalloc((void**)&kernel_device, kernel_size.width * kernel_size.height * sizeof(Npp32s));
	cudaMemcpy(kernel_device, kernel_host, kernel_size.width * kernel_size.height * sizeof(Npp32s), cudaMemcpyHostToDevice);
	NppiPoint anchor = {kernel_size.width/2, kernel_size.height/2};
	Npp32s scale = 1;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	NppStatus status_npp;
	status_npp = nppiFilter_8u_C1R(image_device.data(), image_device.pitch(),
																				filtered_device.data(), filtered_device.pitch(),
																				roi, kernel_device, kernel_size, anchor, scale);

	std::chrono::high_resolution_clock::time_point end_no_transfer = std::chrono::high_resolution_clock::now();
	
	std::cout << "NppiFilter error status " << status_npp << std::endl;
	filtered_device.copyTo(filtered_image.data(), filtered_image.pitch());
	
	std::chrono::high_resolution_clock::time_point end_w_transfer = std::chrono::high_resolution_clock::now();

	saveImage(output_filename, filtered_image);
	
	std::chrono::duration<double> diff_no_transfer = end_no_transfer - start;
	std::chrono::duration<double> diff_w_transfer = end_w_transfer - start;
	// Print times
	std::cout << "Time to process without transfer [" << image_host.width() << "x" << image_host.height() << "] samples (" << 1000*diff_no_transfer.count() << ") ms\n";
	std::cout << "Time to process with transfer [" << image_host.width() << "x" << image_host.height() << "] samples {" << 1000*diff_w_transfer.count() << "} ms\n";

  nppiFree(filtered_device.data());
	nppiFree(image_device.data());
	return EXIT_SUCCESS;
}
