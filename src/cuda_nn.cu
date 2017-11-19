// Compile using nvcc <file> -lcublas -o <output>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define block size for thread allocation
#define BLOCK_DIM 32
#define N 10

/**
 * Define structures used in functions.
 */
typedef struct _kernelParams
{
    int block_size;
    int grid_size;
} sKernelParams;

typedef struct _matrixSize // Optional Command-line multiplier for matrix sizes
{
    unsigned int A_height, A_width, B_height, B_width, C_height, C_width;
} MatrixSize;

typedef struct _vSize // Optional Command-line multiplier for matrix sizes
{
    unsigned int len_A, len_B, len_C;
} VectorSize;

//======================================================================================================================
//=== Structure functions
//======================================================================================================================

/**
 * @brief -  sets values of vector size structure
 *
 * @param vector_size - pointer to vector size struct
 * @param len - length of all vectors
 */
void SetVectorSize(VectorSize *vector_size, unsigned int &len)
{
    vector_size->len_A = len;
    vector_size->len_B = len;
    vector_size->len_C = len;

    printf("Vecotor A(%u), Vecotor B(%u), Vecotor (%u)\n",
           vector_size->len_A,
           vector_size->len_B,
           vector_size->len_C);///////////////////////////

    if (vector_size->len_A != vector_size->len_B ||
        vector_size->len_B != vector_size->len_C ||
        vector_size->len_C != vector_size->len_A)
    {
        printf("ERROR: Matrix sizes do not match!\n");
        exit(-1);
    }
}

/**
 * @brief -  sets values of matrix size structure
 *
 * @param matrixSize - reference to matrix size struct
 * @param widthA - width of matrix A
 * @param heightA - height of matrix A
 * @param widthB - width of matrix B
 * @param heightB - height of matrix B
 * @param widthC - width of matrix C
 * @param heightC - height of matrix C
 */
void SetMatrixSize(MatrixSize *matrixSize,
                   unsigned int widthA, unsigned int heightA,
                   unsigned int widthB, unsigned int heightB,
                   unsigned int widthC, unsigned int heightC)
{
    matrixSize->A_height = heightA;
    matrixSize->A_width = widthA;
    matrixSize->B_height = heightB;
    matrixSize->B_width = widthB;
    matrixSize->C_height = heightC;
    matrixSize->C_width = widthC;
}


//======================================================================================================================
//=== GPU memory initialization functions
//======================================================================================================================

/**
 * @brief - allocates memory on GPU for vectors A, B, and C then copies the values for vector A and B
 *          from host PC onto the device
 *
 * @param argc - from compiler
 * @param argv - from compiler
 * @param devID - device ID number
 * @param vector_size - reference to vector size structure
 * @param host_vA - pointer to host vector A (with values)
 * @param host_vB - pointer to host vector B (with values)
 * @param dev_A - pointer to vector A device memory reference
 * @param dev_B - pointer to vector B device memory reference
 * @param dev_C - pointer to vector C device memory reference
 */
void VectorInitCUDA(int argc, char **argv, int devID, VectorSize *vector_size, float *host_vA, float *host_vB,
                    float *dev_A, float *dev_B, float *dev_C)
{
    // Assign CUDA variables
    devID = 0;
    cudaGetDevice(&devID);
    cudaError_t err;

    // Assign size variables
    size_t size_A = vector_size->len_A * sizeof(float);
    size_t size_B = vector_size->len_B * sizeof(float);
    size_t size_C = vector_size->len_C * sizeof(float);

    // Allocate memory on GPU
    err = cudaMalloc((void **) &dev_A, size_A);
    if (err != cudaSuccess) printf("Allocate vector A: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_B, size_B);
    if (err != cudaSuccess) printf("Allocate vector B: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_C, size_C);
    if (err != cudaSuccess) printf("Allocate vector C: %s\n", cudaGetErrorString(err));

    // Copy data from host PC to GPU
    err = cudaMemcpy(dev_A, host_vA, size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Copy vector A to GPU: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(dev_B, host_vB, size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Copy vector B to GPU: %s\n", cudaGetErrorString(err));

}

/**
 * @brief - allocates memory on GPU for matrices A, B, and C then copies the values for matrices A, B and C
 *          from host PC onto the device
 *
 * @param argc - from compiler
 * @param argv - from compiler
 * @param devID - device ID number
 * @param matrixSize - reference to vector size structure
 * @param host_matrixA - pointer to host matrix A (with values)
 * @param host_matrixB - pointer to host matrix B (with values)
 * @param host_matrixC - pointer to host matrix C (with values)
 * @param dev_matrixA - pointer to matrix A device memory reference
 * @param dev_matrixB - pointer to matrix B device memory reference
 * @param dev_matrixC - pointer to matrix C device memory reference
 */
void MatrixInitCUDA(int argc, char **argv, int &devID, MatrixSize *matrixSize,
                    float *host_matrixA, float *host_matrixB, float *host_matrixC,
                    float *dev_matrixA, float *dev_matrixB, float *dev_matrixC)
{
    // Assign CUDA variables
    devID = 0;
    cudaGetDevice(&devID);
    cudaError_t err;

    // Assign size variables
    size_t matrixA_size = matrixSize->A_height * matrixSize->A_width * sizeof(float);
    printf("Allocation size: %d\tMatrix Size: %d\n", (int) matrixA_size, matrixSize->A_height * matrixSize->A_width);
    size_t matrixB_size = matrixSize->B_height * matrixSize->B_width * sizeof(float);
    size_t matrixC_size = matrixSize->C_height * matrixSize->C_width * sizeof(float);

    // Allocate memory on GPU
    err = cudaMalloc((void **) &dev_matrixA, matrixA_size);
    printf("DEV A POST ALLOC: %p\n", dev_matrixA);
    if (err != cudaSuccess) printf("Allocate matrix A: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_matrixB, matrixB_size);
    if (err != cudaSuccess) printf("Allocate matrix B: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_matrixC, matrixC_size);
    if (err != cudaSuccess) printf("Allocate matrix C: %s\n", cudaGetErrorString(err));

    // Copy data from host PC to GPU
    err = cudaMemcpy(dev_matrixA, host_matrixA, matrixA_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Copy matrix A to GPU: %s\n", cudaGetErrorString(err));
    err =cudaMemcpy(dev_matrixB, host_matrixB, matrixB_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Copy matrix B to GPU: %s\n", cudaGetErrorString(err));
    err =cudaMemcpy(dev_matrixC, host_matrixC, matrixC_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Copy matrix C to GPU: %s\n", cudaGetErrorString(err));
}

//======================================================================================================================
//=== CUDA Vector Functions
//======================================================================================================================

//======================================================================================================================
//=== CUDA Matrix Functions
//======================================================================================================================

/**
 * @brief -  Uses CUBLAS library to perform alpha(A x B) + beta(C) matrix multiplication and addition
 * @param argc - from compiler
 * @param argv - from compiler
 * @param devID - device ID number
 * @param matrixSize - reference to vector size structure
 * @param host_matrixA - pointer to host matrix A (with values)
 * @param host_matrixB - pointer to host matrix B (with values)
 * @param host_matrixC - pointer to host matrix C (with values)
 * @param alpha - value for alpha in CUBLAS function
 * @param beta - value for beta in CUBLAS function
 * @param transposeA - true if A should be transposed
 * @param transposeB - true if B should be transposed
 */

void MatrixMultiply(int argc, char **argv, int &devID, MatrixSize *matrixSize,
                    float *host_matrixA, float *host_matrixB, float *host_matrixC,
                    float alpha, float beta, bool transposeA, bool transposeB)
{
    // Assign CUDA variables
    devID = 0;
    cublasHandle_t handle;
    cudaError_t err;
    cudaGetDevice(&devID);
    cublasCreate(&handle);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devID);
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(matrixSize->C_width / threads.x, matrixSize->C_height/ threads.y);

    // Assign computation variables
    float *dev_matrixA = NULL, *dev_matrixB = NULL, *dev_matrixC = NULL;
    int m = matrixSize->A_height;
    int n = matrixSize->B_width;
    int k = matrixSize->A_width;
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    if (transposeA) transA = CUBLAS_OP_T;
    if (transposeB) transB = CUBLAS_OP_T;
    size_t matrixC_size = matrixSize->C_height * matrixSize->C_width * sizeof(float);

    printf("DEV A PRE ALLOC: %p\n", dev_matrixA);

    // Initialize memory on GPU
    MatrixInitCUDA(argc, argv, devID, matrixSize,
                   host_matrixA, host_matrixB, host_matrixC,
                   dev_matrixA, dev_matrixB, dev_matrixC);

    printf("DEV A POST INIT: %p\n", dev_matrixA);

    // Perform matrix multiplication
    // SGEMM PARAMS: (handle, transposeA, transposeB, m, n, k, alpha, matrix A, k, matrix B, n, beta, matrix C, n)

    cublasSgemm(handle, transA, transB, m, n, k, &alpha, dev_matrixA, k,
                dev_matrixB, n, &beta, dev_matrixC, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("SGEMM: %s\n", cudaGetErrorString(err));

    // Make sure device is finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Device synchronize: %s\n", cudaGetErrorString(err));

    // Copy data from GPU to host PC
    err = cudaMemcpy(host_matrixC, dev_matrixC, matrixC_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("Copy matrix C to Host: %s\n", cudaGetErrorString(err));

    // Free GPU memory
    err = cudaFree(dev_matrixA);
    if (err != cudaSuccess) printf("Free matrix A on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixB);
    if (err != cudaSuccess) printf("Free matrix B on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixC);
    if (err != cudaSuccess) printf("Free matrix C on GPU: %s\n", cudaGetErrorString(err));
}

__global__ void MatrixHadamardKernel (float *dev_matrixA, float *dev_matrixB, float *dev_matrixC,
                                      int C_width, int C_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + row * N;
    if (col < N && row < N)
    {
        dev_matrixC[index] = dev_matrixA[index] * dev_matrixB[index];
    }
}

void ComputeMatrixHadamard(int argc, char **argv, int &devID, MatrixSize *matrixSize,
                           float *host_matrixA, float *host_matrixB, float *host_matrixC)
{
    // Assign CUDA variables
    cudaError_t err;
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    //unsigned int gridX = (unsigned int)ceil(matrixSize->C_width/threads.x);
    //unsigned int gridY = (unsigned int)ceil(matrixSize->C_height/threads.x);
    dim3 grid((int)ceil(N/threads.x),(int)ceil(N/threads.y));

    // Assign computation variables
    float *dev_matrixA = NULL, *dev_matrixB = NULL, *dev_matrixC = NULL;
    size_t matrixC_size = matrixSize->C_height * matrixSize->C_width * sizeof(float);

    // Initialize memory on GPU
    MatrixInitCUDA(argc, argv, devID, matrixSize,
                   host_matrixA, host_matrixB, host_matrixC,
                   dev_matrixA, dev_matrixB, dev_matrixC);


    // Compute Hadamard Product
    MatrixHadamardKernel<<<grid,threads>>>(dev_matrixA,dev_matrixB,dev_matrixC, N, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Hadamard Computation: %s\n", cudaGetErrorString(err));

    // Make sure device is finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Device synchronize: %s\n", cudaGetErrorString(err));

    // Copy data from GPU to host PC
    err = cudaMemcpy(host_matrixC, dev_matrixC, matrixC_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("Copy matrix C to Host: %s\n", cudaGetErrorString(err));

    // Free GPU memory
    err = cudaFree(dev_matrixA);
    if (err != cudaSuccess) printf("Free matrix A on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixB);
    if (err != cudaSuccess) printf("Free matrix B on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixC);
    if (err != cudaSuccess) printf("Free matrix C on GPU: %s\n", cudaGetErrorString(err));

}

//======================================================================================================================
//=== Main Function
//======================================================================================================================

/**
 * @brief - computes weight matrices for a shallow neural network
 * @param argc - from compiler
 * @param argv - from compiler
 * @return 0 if success
 */
int main(int argc, char **argv)
{
    // Create memory for Layer 1, Layer 2, Layer 3 vectors
    // float *layer1 = malloc(784*sizeof(floats)))
    // Create memory for Weight 1->2, Weight 2->3 matrices

    // Layer 1 will read from file for input (X) values
    // Layer 2 and 3 will be calculated
    int devID = 0;
    cudaGetDevice(&devID);

    // Testing hadamard product, init function, and set matrix size function
    float *host_A, *host_B, *host_C;
    MatrixSize *hadamardTest = (MatrixSize*) calloc(sizeof(MatrixSize), 1);
    size_t calcSize = N*N*sizeof(float);
    host_A = (float *)calloc(calcSize, 1);
    host_B = (float *)calloc(calcSize, 1);
    host_C = (float *)calloc(calcSize, 1);
    SetMatrixSize(hadamardTest, N, N, N, N, N, N);


    for(int i = 0; i < N*N; i ++)
    {
        host_A[i] = 2.0;
        host_B[i] = 7.0;
    }

    printf("Matrix A:\n");
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f ", host_A[i*j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");

    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f ", host_B[i*j]);
        }
        printf("\n");
    }
    printf("\n");
    //ComputeMatrixHadamard(argc, argv, devID, hadamardTest, host_A, host_B, host_C);

    MatrixMultiply(argc, argv, devID, hadamardTest, host_A, host_B, host_C, 1, 1, false, false);

    printf("\nMatrix C:\n");
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%f ", host_C[i*j]);
        }
        printf("\n");
    }

    return 0;
}
