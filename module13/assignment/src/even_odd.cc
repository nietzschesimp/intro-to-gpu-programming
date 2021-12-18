#include "argument_parser.h"
#include "opencl_utils.h"
#include <csignal>

#define DEFAULT_PLATFORM 0


bool exit_program = false;

struct iodata {
    char* operation;
    int* in;
    int* classification;
    size_t in_size;
};


void CL_CALLBACK event_callback(cl_event event, cl_int cmd_status, void* user_data)
{
    iodata* data = (iodata*)user_data;
    std::cout << "[NUM]:\t\t[CLASSIFICATION: " << data->operation<< "]\n";
    for (int ii = 0; ii < data->in_size; ii++) {
        std::cout << data->in[ii] << "\t\t" << ((data->classification[ii] == 1) ? "true": "false") << std::endl;
    }
    exit_program = true;
}


void CL_CALLBACK start_event_callback(cl_event event, cl_int cmd_status, void* user_data)
{
    std::cout << "Processing event: " << (char*)(user_data) << std::endl;
}


void sigint_handler(int signal) 
{
    std::cout << "Exiting program...\n";
    exit_program = true;
}


int main(int argc, char** argv)
{
    cl_platform_id platforms;
    cl_context context[2];
    cl_command_queue queues[2];
    cl_kernel kernels[2];
    cl_mem buffer[2];
    cl_event start_event[2], copy_event[2], proc_event[2], complete[2];
    int platform_default = 0;
    std::string input_filename;
    std::string kernel_names[2] = {"is_even", "is_odd"};
    std::string mode;
    std::string source = "even_odd.cl";

    // Read command line arguments
    ArgumentParser parser(argc, argv);
	  if (parser.exists("-i")) {
	  		input_filename = parser.get_option("-i");
	  }
	  if (parser.exists("-m")) {
	  	 mode = parser.get_option("-m");
	  }
	  if (parser.exists("-s")) {
	  	 source = parser.get_option("-s");
	  }
    
    // Read .cl file
    std::ifstream src_file(source, std::ios::in);
    if (!src_file.is_open()) {
        std::cerr << "Failed to open file for reading: even_odd.cl\n";
    }
    std::ostringstream oss;
    oss << src_file.rdbuf();
    std::string src_prog = oss.str();
    
    // Read Input file
    std::vector<int> in_numbers;
    int x;
    std::cout << "Opening: " << input_filename <<std::endl;
    std::ifstream in_file(input_filename, std::ios::in);
    if (!in_file.is_open()) {
      throw std::runtime_error("Failed to open file for reading: " + input_filename);
    }
    while (in_file >> x) {
      in_numbers.push_back(x);
    }
    std::cout << "Read input files" << std::endl;
    
    // Space for kernel output
    int out[in_numbers.size()];
    const size_t globalWorkSize[1] = {in_numbers.size()};
		const size_t localWorkSize[1] = {1};

    // Find OpenCL platforms
    int err = find_platform(&platforms, 1);

    for (int idx = 0; idx < 2; idx++) {
      // Create contexts for each queue
      err |= create_context(context[idx], &platforms, DEFAULT_PLATFORM);

      // Create command queues
      err |= create_command_queue(queues[idx], context[idx]);

      // Build Kernel
      err |= build_kernel(kernels[idx], context[idx], src_prog, kernel_names[idx]);
      
      // Create buffer in device
      err |= create_buffer(&buffer[idx], &context[idx], CL_MEM_READ_WRITE, sizeof(int)*in_numbers.size());

      // Create start event in host
      start_event[idx] = clCreateUserEvent(context[idx], NULL);
      clSetEventCallback(start_event[idx], CL_RUNNING, &start_event_callback, (void*)kernel_names[idx].c_str());
    
      // Copy input to device asynchronously
      err |= clEnqueueWriteBuffer(queues[idx],
                                 buffer[idx],
                                 CL_FALSE,
                                 0,
                                 sizeof(int) * in_numbers.size(),
                                 in_numbers.data(),
                                 1,
                                 &start_event[idx],
                                 &copy_event[idx]);
      clSetEventCallback(copy_event[idx], CL_RUNNING, &start_event_callback, (void*)"copy_event");

      // Enqueue execution kernel to run after input is copied 
      err |= clSetKernelArg(kernels[idx], 0, sizeof(cl_mem), &buffer[idx]);
      err |= clSetKernelArg(kernels[idx], 1, sizeof(cl_mem), &buffer[idx]);
      err |= clEnqueueNDRangeKernel(queues[idx],
                                    kernels[idx], 
                                    1, 
                                    NULL,
                                    globalWorkSize, 
                                    localWorkSize, 
                                    1,
                                    &copy_event[idx],
                                    &proc_event[idx]);
      clSetEventCallback(proc_event[idx], CL_RUNNING, &start_event_callback, (void*)"proc_event");

      // Copy results into host asynchronously after finishing processing the samples
      err |= clEnqueueReadBuffer(queues[idx], 
                                 buffer[idx], 
                                 CL_FALSE,
                                 0, 
                                 sizeof(int) * in_numbers.size(),
                                 out,
                                 1,
                                 &proc_event[idx],
                                 &complete[idx]);
    }
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating OpenCL objects.\n";
        return 1;
    }
    std::cout << "Allocated OpenCL objects\n";

    // Set data for callback
    iodata dat;
    dat.operation = const_cast<char*>(mode.c_str());
    dat.in = in_numbers.data();
    dat.in_size = in_numbers.size();
    dat.classification = out;

    // Trigger analysis on device based on input
    int option = -1;
    if (mode == "even") {
        option = 0;
    }
    else if (mode == "odd") {
        option = 1;
    }
    else {
        std::cout << "[ERROR] Invalid operation mode\n";
        return 1;
    }
    
    clSetEventCallback(complete[option], CL_COMPLETE, &event_callback, (void*)(&dat));
    clSetUserEventStatus(start_event[option], CL_SUCCESS);

    // In here the CPU is free to do stuff, I will place a wait to simulate busywork
    while (not exit_program) {
        continue;
    }

    // Calculate time
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(start_event[option], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	  clGetEventProfilingInfo(complete[option], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	  double delta = time_end - time_start;
    std::cout << "Time to process input: {" << delta/1e6 << "} ms\n";

    return 0;
}
