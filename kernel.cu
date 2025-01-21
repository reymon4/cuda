/*
__device__
int aux(int x)
{
    return x*2;
}
*/
#include <iostream>
__global__
void suma_kernel(float* a, float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x+threadIdx.x ;
    //aux(a[0]);
    if (index < size)
        c[index] = a[index] + b[index];
}
//Esta función puedo invocar en el archivo cpp
extern "C"
void suma_paralela(float* a, float* b, float* c, int size) {
    int thread_num = 1024;
    int block_num = std::ceil(size/ (float) thread_num);
    std::printf("num_blocks: %d, num_threads: %d\n", block_num, thread_num);
    suma_kernel<<<block_num, thread_num>>>(a, b, c, size);
}