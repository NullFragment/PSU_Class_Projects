// Compile using nvcc <file> -lcublas -o <output>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define block size for thread allocation
// #define block_size = 1024

typedef struct _kernelParams
{
    int block_size;
    int grid_size;
} sKernelParams;

typedef struct _matrixSize // Optional Command-line multiplier for matrix sizes
{
    unsigned int A_height, A_width, B_height, B_width, C_height, C_width;
} sMatrixSize;

typedef struct _vSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int len_A, len_B, len_C;
} VectorSize;







void SetVectorSize(int &len, VectorSize &vector_size)
{
    vector_size.len_A = len;
    vector_size.len_B = len;
    vector_size.len_C = len;

    printf("Vecotor A(%u), Vecotor B(%u), Vecotor (%u)\n",
           vector_size.len_A,
           vector_size.len_B,
           vector_size.len_C);///////////////////////////

    if (vector_size.len_A != vector_size.len_B ||
        vector_size.len_B != vector_size.len_C ||
        vector_size.len_C != vector_size.len_A)
    {
        printf("ERROR: Matrix sizes do not match!\n");
        exit(-1);
    }
}

void SetMatrixSize(sMatrixSize *matrixSize, widthA, heightA, widthB, heightB, widthC, heightC)
{
    matrixSize->A_height = heightA;
    matrixSize->A_width = widthA;
    matrixSize->B_height = heightB;
    matrixSize->B_width = widthB;
    matrixSize->C_height = heightC;
    matrixSize->C_width = widthC;
}








void VectorInitCUDA(int argc, char **argv, int devID, VectorSize &vector_size, float *host_vA, float *host_vB,
                    float *host_vC, float *dev_A, float *dev_B, float *dev_C)
{

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

void MatrixInitCUDA(int argc, char **argv, int &devID, sMatrixSize &matrixSize,
                    float *host_matrixA, float *host_matrixB, float *host_matrixC,
                    float *dev_matrixA, float *dev_matrixB, float *dev_matrixC)
{
    devID = 0;
    cudaGetDevice(&devID);
    int matrixA_size = matrixSize.A_height * matrixSize.A_width * sizeof(float);
    int matrixB_size = matrixSize.B_height * matrixSize.B_width * sizeof(float);
    int matrixC_size = matrixSize.C_height * matrixSize.C_width * sizeof(float);
    cudaMalloc((void **) &dev_matrixA, matrixA_size);
    cudaMalloc((void **) &dev_matrixB, matrixB_size);
    cudaMalloc((void **) &dev_matrixC, matrixC_size);
    cudaMemcpy(dev_matrixA, host_matrixA, matrixA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixA, host_matrixA, matrixB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixA, host_matrixA, matrixB_size, cudaMemcpyHostToDevice);
}








void matrixMultiply(int argc, char **argv, int &devID, sMatrixSize &matrixSize,
                    float *host_matrixA, float *host_matrixB, float *host_matrixC,
                    float alpha, float beta, bool transposeA, bool transposeB)
{
    devID = 0;
    cudaGetDevice(&devID);
    cublasCreate(&handle);
    float *dev_matrixA, *dev_matrixB, *dev_matrixc;
    int m = matrixSize.A_height;
    int n = matrixSize.B_width;
    int k = matrixSize.A_width;
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    if (transposeA) transA = CUBLAS_OP_T;
    if (transposeB) transB = CUBLAS_OP_T;

    // Initialize memory on device
    MatrixInitCUDA(argc, argv, devID, matrixSize,
                   host_matrixA, host_matrixB, host_matrixC,
                   dev_matrixA, dev_matrixB, dev_matrixC);

    // Perform matrix multiplication
    // SGEMM PARAMS:
    // (handle, transposeA, transposeB, m, n, k, alpha, matrix A, k, matrix B, n, beta, matrix C, n)
    cublasSgemm(handle, transA, transB, m, n, k, &alpha, dev_matrixA, k,
                dev_matrixB, n, &beta, dev_matrixC, n);
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixC);
}

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
    cudaGetDevice(&devID);
}
