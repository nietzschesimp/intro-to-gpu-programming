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


void checkErr(cl_int err, const char * name);
int find_platform(cl_platform_id* platform_ids, int num_platforms);
int create_context(cl_context& context, cl_platform_id* platformIDs, int platform);
int create_buffer(cl_mem* buffer, cl_context* context, cl_int flags, size_t num_bytes);
int build_kernel(cl_kernel& kernel, cl_context& context, std::string source, std::string kernel_name);
int create_command_queue(cl_command_queue& queue, cl_context& context);
