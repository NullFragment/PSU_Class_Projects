// Compile using nvcc <file> -lcublas -o <output>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define block size for thread allocation
// #define block_size = 1024

typedef struct _kernelParams {
        int block_size;
        int grid_size;
} sKernelParams;

typedef struct _matrixSize // Optional Command-line multiplier for matrix sizes
{
        unsigned int A_height, A_width, B_height, B_width, C_height, C_width;
} sMatrixSize;

/**
   This function will initialize memory on GPU for matrices and copy values
   for matrix operation
 */
void MatrixInitCUDA(int argc, char **argv, int &devID, sMatrixSize &matrixSize,
                    float *host_matrixA, float *host_matrixB, float *host_matrixC,
                    float *host_alpha, float *host_beta,
                    float *dev_matrixA, float *dev_matrixB, float *dev_matrixC,
                    float *dev_alpha, float *dev_beta) {
        devID = 0;
        cudaGetDevice(&devID);
        int matrixA_size = matrixSize.A_height * matrixSize.A_width * sizeof(float);
        int matrixB_size = matrixSize.B_height * matrixSize.B_width * sizeof(float);
        int matrixC_size = matrixSize.C_height * matrixSize.C_width * sizeof(float);
        cudaMalloc((void **)&dev_matrixA, matrixA_size);
        cudaMalloc((void **)&dev_matrixB, matrixB_size);
        cudaMalloc((void **)&dev_matrixC, matrixC_size);
        cudaMemcpy(dev_matrixA, host_matrixA, matrixA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_matrixA, host_matrixA, matrixB_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_matrixA, host_matrixA, matrixB_size, cudaMemcpyHostToDevice);
}

/**
   This function will initialize memory on GPU for vectors and copy values
   for vector operations
 */
// void VectorInitCUDA

/**
   This function will call the initialize function then call
   cublasSgemm function, copy result back to host and free GPU memory
 */
// float matrixMultiply
// cudaFree(dev_matrixA);
// cudaFree(dev_matrixB);
// cudaFree(dev_matrixC);



/**
   This function will call vector init function,
   perform vector multiplication, then free GPU memory
 */
// float vectorMultiply

int main(int argc, char **argv)
{
        // Create memory for Layer 1, Layer 2, Layer 3 vectors
        // float *layer1 = malloc(784*sizeof(floats)))
        // Create memory for Weight 1->2, Weight 2->3 matrices

        // Layer 1 will read from file for input (X) values
        // Layer 2 and 3 will be calculated
        int devID = 0;
        sMatrixSize activation2_size;
        activation2_size.A_width = 128;
        activation2_size.A_height = 128;
        activation2_size.B_width = 128;
        activation2_size.B_height = 128;
        activation2_size.C_width = 128;
        activation2_size.C_height = 128;
        MatrixInitCUDA(argc, argv, devID, activation2_size);
}
