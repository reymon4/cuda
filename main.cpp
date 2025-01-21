//
// Created by fing.labcom on 21/01/2025.
//
#include <iostream>
#include <cuda_runtime.h>


const size_t VECTOR_SIZE =1024*1024;

void suma_serial(float* x, float* y, float* z, int size)
{
    for(int i=0; i<size; i++)
    {
        z[i] = x[i] + y[i];
    }
}

extern "C"
void suma_paralela(float* a, float* b, float* c, int size);
int main()
{
    // Allocate memory for the host vectors
    float* h_a= new float[VECTOR_SIZE];
    float* h_b= new float[VECTOR_SIZE];
    float* h_c = new float[VECTOR_SIZE];

    memset(h_c, 0, VECTOR_SIZE * sizeof(float));

    for(int i=0; i<VECTOR_SIZE; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate memory for the device vectors
    float* d_a;
    float* d_b;
    float* d_c;

    size_t size = VECTOR_SIZE * sizeof(float);

    //std::printf("%zd",size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy the host vectors to the device
    //destino, valor, tamaño, dirección
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    //10 grids de 256 bloques
    suma_paralela(d_a, d_b, d_c, VECTOR_SIZE);
    //Copy the device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //Print results
    std::printf("\n\n resultado: ");
    for(int i=0; i<10;i++)
    {
        std::printf("%.0f", h_c[i]);
    }

    return 0;
}
