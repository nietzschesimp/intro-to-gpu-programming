//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

#include "argument_parser.h"


// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_uint inputSignal[inputSignalHeight][inputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_uint mask[maskHeight][maskWidth] =
{
	{0, 0, 1, 2, 1, 0, 0},
	{0, 1, 2, 3, 2, 1, 0},
	{1, 2, 3, 4, 3, 2, 1},
	{2, 3, 4, 0, 4, 3, 2},
	{1, 2, 3, 4, 3, 2, 1},
	{0, 1, 2, 3, 2, 1, 0},
	{0, 0, 1, 2, 1, 0, 0},
};

const unsigned int outputSignalWidth  = inputSignalWidth - 2*(maskWidth/2);
const unsigned int outputSignalHeight = inputSignalHeight - 2*(maskHeight/2);

cl_uint outputSignal[outputSignalHeight][outputSignalWidth];


// Function to check and handle OpenCL errors
inline void checkErr(cl_int err, const char * name)
{
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CL_CALLBACK contextCallback(const char * errInfo, const void * private_info,
	                      size_t cb, void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

void fillMatrixRand(cl_uint* dest, unsigned int width, unsigned int height)
{
	std::srand(std::time(nullptr));
	for (int ii=0; ii < width; ii++) 
	{
		for (int yy=0; yy < height; yy++)
		{
			dest[ii*width + yy] = std::rand() % 4;
		}
	}
}

cl_program createProgram(const char* filename,
									 cl_context& context,
									 cl_device_id* deviceIDs,
									 cl_uint& numDevices)
{
  cl_int errNum;
	std::ifstream srcFile(filename);
	std::string error_info = "reading ";
	error_info += filename;
  checkErr(srcFile.is_open() ? CL_SUCCESS : -1, error_info.c_str());

	std::string srcProg(std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>())
	);

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	cl_program program = clCreateProgramWithSource(context, 
																								 1, 
																								 &src, 
	                  														 &length,
																								 &errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(program, 
													numDevices, 
													deviceIDs, 
													NULL, 
													NULL, 
													NULL);
  if (errNum != CL_SUCCESS)
  {
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, 
													deviceIDs[0], 
													CL_PROGRAM_BUILD_LOG,
				 									sizeof(buildLog), 
													buildLog, 
													NULL);

    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << buildLog;

		checkErr(errNum, "clBuildProgram");
  }
	return program;
}


//	main() for Convoloution example
int main(int argc, char** argv)
{
  cl_int errNum;
  cl_uint numPlatforms;
	cl_uint numDevices;
  cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
  cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel, scaling;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;
	cl_event conv_event, scale_event;
	cl_ulong time_start, time_end;

	std::string convolution_filename = "Convolution.cl";
	std::string scale_filename = "Scaling.cl";

  // read command line arguments
  ArgumentParser parser(argc, argv);
	if (parser.exists("-s")) {
			scale_filename = parser.get_option("-s");
	}
	if (parser.exists("-c")) {
		 convolution_filename = parser.get_option("-c");
	}

  // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, 
														NULL, 
														&numPlatforms);
	checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
						"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

  errNum = clGetPlatformIDs(numPlatforms,
														platformIDs, 
														NULL);
  checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   				"clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(platformIDs[i],
														CL_DEVICE_TYPE_GPU, 
                  					0, 
														NULL, 
														&numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	  {
			checkErr(errNum, "clGetDeviceIDs");
    }
	  else if (numDevices > 0) 
	  {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			  errNum = clGetDeviceIDs(platformIDs[i],
																CL_DEVICE_TYPE_GPU,
				              					numDevices, 
																&deviceIDs[0], 
																NULL);
			  checkErr(errNum, "clGetDeviceIDs");
			  break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
  // 	if (deviceIDs == NULL) {
  // 		std::cout << "No CPU device found" << std::endl;
  // 		exit(-1);
  // 	}

  // Next, create an OpenCL context on the selected platform.  
  cl_context_properties contextProperties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platformIDs[i],
    0
  };
  context = clCreateContext(contextProperties, 
														numDevices, 
														deviceIDs, 
	                  				&contextCallback,
														NULL,
														&errNum);
	checkErr(errNum, "clCreateContext");

	// Create program from source
	program = createProgram(convolution_filename.c_str(), 
													context, 
													deviceIDs, 
													numDevices);

	// Create kernel object
	kernel = clCreateKernel(program, "convolve", &errNum);
	checkErr(errNum, "clCreateKernel(convolution)");

	// Create input buffer
	fillMatrixRand(&inputSignal[0][0], inputSignalHeight, inputSignalWidth);	
	inputSignalBuffer = clCreateBuffer(context, 
																		 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
																		 sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
																		 static_cast<void *>(inputSignal), 
																		 &errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	// Create convolution kernel buffer
	maskBuffer = clCreateBuffer(context, 
															CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
															sizeof(cl_uint) * maskHeight * maskWidth,
															static_cast<void *>(mask), 
															&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	// Create output buffer
	outputSignalBuffer = clCreateBuffer(context, 
																			CL_MEM_WRITE_ONLY,
																			sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
																			NULL,
																			&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(context, 
															 deviceIDs[0], 
															 CL_QUEUE_PROFILING_ENABLE, 
															 &errNum);
	checkErr(errNum, "clCreateCommandQueue");

	// Set kernel arguments
  errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg(convolution)");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
  const size_t localWorkSize[2]  = { 1, 1 };

	// Create scaling program
	program = createProgram(scale_filename.c_str(),
													context, 
													deviceIDs, 
													numDevices);

	// Create scaling kernel
	scaling = clCreateKernel(program, "scale", &errNum);
	checkErr(errNum, "clCreateKernel(scaling)");
	cl_uint scaleFactor = 4;

	// Set scaling arguments
  errNum  = clSetKernelArg(scaling, 0, sizeof(cl_mem), &outputSignalBuffer);
  errNum |= clSetKernelArg(scaling, 1, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(scaling, 2, sizeof(cl_uint), &scaleFactor);
	checkErr(errNum, "clSetKernelArg(scaling)");

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue,
																	kernel, 
																	2, 
																	NULL,
			              							globalWorkSize, 
																	localWorkSize, 
																	0,
																	NULL,
																	&conv_event);
	checkErr(errNum, "clEnqueueNDRangeKernel(convolution)");

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, 
																	scaling, 
																	2, 
																	NULL,
			              							globalWorkSize, 
																	localWorkSize, 
																	0, 
																	NULL, 
																	&scale_event);
	checkErr(errNum, "clEnqueueNDRangeKernel(scaling)");

	// Read results from device into host
	errNum = clEnqueueReadBuffer(queue, 
															 outputSignalBuffer, 
															 CL_TRUE, 
															 0, 
															 sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
															 outputSignal, 
															 0, 
															 NULL, 
															 NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

  // Output the result buffer
  for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

  std::cout << std::endl << "Executed program succesfully." << std::endl;

	clGetEventProfilingInfo(conv_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(conv_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	double delta = time_end-time_start;

	clGetEventProfilingInfo(scale_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(scale_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	delta += time_end-time_start;

  std::cout << "Time to process an [" << outputSignalWidth << "][" << outputSignalHeight<< "] samples {" << delta/1e6 << "} ms\n";

	return 0;
}
