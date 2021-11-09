#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "argument_parser.h"


///
//  Constants
//
const int ARRAY_SIZE_DEFAULT = 1000;

///
//  Create a random float 
///
int randrange(int val_min, int val_max) {
	return val_min + (std::rand() % (val_max - val_min + 1));
}

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context create_context()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
            NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue create_command_queue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program create_program(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
            (const char**)&srcStr,
            NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool create_mem_objects(cl_context context, cl_mem memObjects[3], float *a, float *b, int array_size)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * array_size, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * array_size, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(float) * array_size, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void cleanup(cl_context context, cl_command_queue commandQueue,
        cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    size_t array_size = ARRAY_SIZE_DEFAULT;
    std::string kernel_filename = "vector_operations.cl";
    std::string op = "";
    // read command line arguments
    ArgumentParser parser(argc, argv);
    if (parser.exists("-s")) {
        array_size = std::stoi(parser.get_option("-s"));
    }
    if (parser.exists("-o")) {
        op = parser.get_option("-o");
    }
    else {
        std::cout << "[ERROR]: No operation indicated.\n";
        return EXIT_FAILURE;
    }
    if (parser.exists("-k")) {
        kernel_filename = parser.get_option("-k");
    }

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = create_context();
    if (context == NULL) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = create_command_queue(context, &device);
    if (commandQueue == NULL) {
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = create_program(context, device, kernel_filename.c_str());
    if (program == NULL) {
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create OpenCL kernel
    std::string op_str = op + "_kernel";
    std::cout << "Kernel: " << op_str << std::endl;
    kernel = clCreateKernel(program, op_str.c_str(), NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create kernel" << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[array_size];
    float a[array_size];
    float b[array_size];
    for (int i = 0; i < array_size; i++) {
        a[i] = (float)i;
        b[i] = (float)randrange(0, 3);
    }

    if (!create_mem_objects(context, memObjects, a, b, array_size)) {
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[2]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[1]);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error setting kernel arguments." << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    size_t globalWorkSize[1] = { array_size };
    size_t localWorkSize[1] = { 1 };

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
            globalWorkSize, localWorkSize,
            0, NULL, NULL
    );
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
            0, array_size * sizeof(float), result,
            0, NULL, NULL
    );
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error reading result buffer." << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Output the result buffer
    std::cout << "a[]:" << "\t" << "b[]:" << "\t" << "output:" << std::endl;
    for (int i = array_size - 10; i < array_size; i++) {
        std::cout << a[i] << "\t" << b[i] << "\t" << result[i] << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    cleanup(context, commandQueue, program, kernel, memObjects);

    std::cout << "Time to process with transfer [" << array_size << "] samples {" << 1000*diff.count() << "} ms\n";

    return EXIT_SUCCESS;
}
