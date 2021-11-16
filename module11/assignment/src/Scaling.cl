
__kernel void scale(__global uint* const result,
                    const __global uint* const input,
                    const uint scale)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  result[y * get_global_size(0) + x] = input[y * get_global_size(0) + x] / scale;
}

