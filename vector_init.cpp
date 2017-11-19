//initialization for vectors
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef struct _vSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int len_A, len_B, len_C;
} VectorSize;

void setVectorSize(int &len, VectorSize &vector_size)
{
    vector_size.len_A = len;
    vector_size.len_B = len;
    vector_size.len_C = len;

    printf("Vecotor A(%u), Vecotor B(%u), Vecotor (%u)\n",
           vector_size.len_A, 
           vector_size.len_B, 
           vector_size.len_C);///////////////////////////

    if( vector_size.len_A  != vector_size.len_B ||
        vector_size.len_B != vector_size.len_C ||
        vector_size.len_C != vector_size.len_A)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

void allocateMem(int argc, char **argv, int devID, VectorSize &vector_size, float *host_vA, float *host_vB, float *host_vC, float *dev_A, float *dev_B, float *dev_C){

	devID = 0;
    cudaGetDevice(&devID);

    int size_A = vector_size.len_A * sizeof(float);
    int size_B = vector_size.len_B * sizeof(float);
    int size_C = vector_size.len_C * sizeof(float);

    cudaMalloc((void **) &dev_A, size_A);
    cudaMalloc((void **) &dev_B, size_B);
    cudaMalloc((void **) &dev_C, size_C);

    cudaMemcpy(dev_A, host_vA, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_vB, size_B, cudaMemcpyHostToDevice);

    return;
}

int main (int argc, char **argv){
	int devID = 0;
	int len_vector = 5;
    VectorSize vector_size;

    setVectorSize(len_vector, vector_size);

    unsigned int size_A = vector_size.len_A;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *dammy1 = (float *)malloc(mem_size_A);

    unsigned int size_B = vector_size.len_B;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *dammy2 = (float *)malloc(mem_size_B);

    unsigned int size_C = vector_size.len_C;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *dammy3 = (float *)malloc(mem_size_C);

    float *dev_A;
    float *dev_B;
    float *dev_C;
    allocateMem(argc, argv, devID, vector_size, dammy1, dammy2, dammy3, dev_A, dev_B, dev_C);
    


    return 0;

}












