add_executable(caesar caesar.cu)
set_property(TARGET caesar PROPERTY CUDA_STANDARD 11)

add_executable(host_memory host_memory.cu)
set_property(TARGET host_memory PROPERTY CUDA_STANDARD 11)

add_executable(pinned_memory pinned_memory.cu)
set_property(TARGET pinned_memory PROPERTY CUDA_STANDARD 11)

add_executable(const_memory const_memory.cu)
set_property(TARGET const_memory PROPERTY CUDA_STANDARD 11)

add_executable(cpu_proc cpu_proc.cu)
set_property(TARGET cpu_proc PROPERTY CUDA_STANDARD 11)

add_executable(shared_memory shared_memory.cu)
set_property(TARGET shared_memory PROPERTY CUDA_STANDARD 11)

add_executable(register_memory register_memory.cu)
set_property(TARGET register_memory PROPERTY CUDA_STANDARD 11)
