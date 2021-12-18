#include "opencl_utils.h"


// Function to check and handle OpenCL errors
void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int find_platform(cl_platform_id* platform_ids, int num_platforms)
{
    cl_int errNum;
    cl_uint numPlatforms;
    
    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    cl_platform_id* platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    for (int i = 0; (i < num_platforms) && (i < numPlatforms); i++)
        platform_ids[i] = platformIDs[i];

    return errNum;
}


int create_context(cl_context& context, cl_platform_id* platformIDs, int platform)
{
    cl_int errNum;
    cl_context ctx = NULL;

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platformIDs[platform]),
        0
    };
    ctx = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return errNum;
        }
    }

   context = ctx;

   return errNum;
}

int create_buffer(cl_mem* buffer, cl_context* context, cl_int flags, size_t num_bytes)
{
    cl_int errNum;
    *buffer = clCreateBuffer((*context),
                             flags,
                             num_bytes,
                             NULL,
                             &errNum);
    checkErr(errNum, "clCreateBuffer");
    return errNum;
}


int build_kernel(cl_kernel& kernel, cl_context& context, std::string source, std::string kernel_name)
{
    cl_int errNum;
    const char* src = source.c_str();
    size_t length = source.length();
    cl_int numDevices = -1;
     
    // Create program from source
    cl_program program = clCreateProgramWithSource(context, 
                                                   1, 
                                                   &src, 
                                                   &length, 
                                                   &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

     // Build program
    errNum = clBuildProgram(program,
                             0,
                             NULL,
                             NULL,
                             NULL,
                             NULL);
 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, 
                              0, 
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), 
                              buildLog, 
                              NULL);

        std::cerr << "Error in OpenCL C source: " << buildLog << std::endl;
        return errNum;
    }
    
    kernel = clCreateKernel(program,
                            kernel_name.c_str(),
                            &errNum);
    std::string err_msg = "clCreateKernel(" + kernel_name + ")";
    checkErr(errNum, err_msg.c_str());
    return errNum;
}


int create_command_queue(cl_command_queue& queue, cl_context& context)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)\n";
        return errNum;
    }

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return errNum;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        delete [] devices;
        std::cerr << "Failed to get device IDs\n";
        return errNum;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &errNum);
    if (errNum != CL_SUCCESS) {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return errNum;
    }

    //*device = devices[0];
    queue = commandQueue;
    delete [] devices;
    return CL_SUCCESS;
}
