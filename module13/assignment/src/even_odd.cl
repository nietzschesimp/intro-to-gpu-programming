__kernel void is_odd(__global int* result, 
                     __global const int* input)
{
  int x = get_global_id(0);
  if ((input[x] % 2) != 0) {
    result[x] = 1;
  }
  else {
    result[x] = 0;
  }
}

__kernel void is_even(__global int* result, 
                      __global const int* input)
{
  int x = get_global_id(0);
  if ((input[x] % 2) == 0) {
    result[x] = 1;
  }
  else {
    result[x] = 0;
  }
}
