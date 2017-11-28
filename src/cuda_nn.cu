// Compile using nvcc <file> -lcublas -o <output>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iostream>

// Define block size for thread allocation
#define NUM_THREADS 32 // 32 is max for N^2 threads: 32*32 = 1024
#define LOGGING 0

//==================================================================================
//=== Structure definitions
//==================================================================================
typedef struct _kernelParams {
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

//==================================================================================
//=== Structure functions
//==================================================================================

/**
 * @brief -  sets values of vector size structure
 *
 * @param vector_size - pointer to vector size struct
 * @param len - length of all vectors
 */
void SetVectorSize(VectorSize *vector_size, unsigned int len) {
    vector_size->len_A = len;
    vector_size->len_B = len;
    vector_size->len_C = len;

    if (LOGGING == 1)
        fprintf(stdout, "Vector A(%u), Vector B(%u), Vector C(%u)\n",
                vector_size->len_A,
                vector_size->len_B,
                vector_size->len_C);

    if (vector_size->len_A != vector_size->len_B ||
        vector_size->len_B != vector_size->len_C ||
        vector_size->len_C != vector_size->len_A) {
        fprintf(stderr, "ERROR: Vector lengths do not match!\n");
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
                   unsigned int widthC, unsigned int heightC) {
    matrixSize->A_height = heightA;
    matrixSize->A_width = widthA;
    matrixSize->B_height = heightB;
    matrixSize->B_width = widthB;
    matrixSize->C_height = heightC;
    matrixSize->C_width = widthC;

    if (LOGGING == 1)
        fprintf(stdout, "Matrix A(%u x %u), Matrix B(%u x %u), Matrix C(%u x %u)\n",
                matrixSize->A_width,
                matrixSize->A_height,
                matrixSize->B_width,
                matrixSize->B_height,
                matrixSize->C_width,
                matrixSize->C_height);

}

//==================================================================================
//=== GPU memory initialization functions
//==================================================================================

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
                    float *&dev_A, float *&dev_B, float *&dev_C) {
    // Assign CUDA variables
    cudaError_t err;

    // Assign size variables
    size_t size_A = vector_size->len_A * sizeof(float);
    size_t size_B = vector_size->len_B * sizeof(float);
    size_t size_C = vector_size->len_C * sizeof(float);

    // Allocate memory on GPU
    err = cudaMalloc((void **) &dev_A, size_A);
    if (err != cudaSuccess) fprintf(stderr, "ERROR allocating vector A: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_B, size_B);
    if (err != cudaSuccess) fprintf(stderr, "ERROR allocating vector B: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_C, size_C);
    if (err != cudaSuccess) fprintf(stderr, "ERROR allocating vector C: %s\n", cudaGetErrorString(err));

    // Copy data from host PC to GPU
    err = cudaMemcpy(dev_A, host_vA, size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying vector A to GPU: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(dev_B, host_vB, size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying vector B to GPU: %s\n", cudaGetErrorString(err));

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
                    float *&dev_matrixA, float *&dev_matrixB, float *&dev_matrixC) {
    // Assign CUDA variables
    cudaError_t err;

    // Assign size variables
    size_t matrixA_size = matrixSize->A_height * matrixSize->A_width * sizeof(float);
    size_t matrixB_size = matrixSize->B_height * matrixSize->B_width * sizeof(float);
    size_t matrixC_size = matrixSize->C_height * matrixSize->C_width * sizeof(float);

    // Allocate memory on GPU
    err = cudaMalloc((void **) &dev_matrixA, matrixA_size);
    if (err != cudaSuccess) fprintf(stderr, "ERROR allocating matrix A: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_matrixB, matrixB_size);
    if (err != cudaSuccess) fprintf(stderr, "ERROR allocating matrix B: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_matrixC, matrixC_size);
    if (err != cudaSuccess) fprintf(stderr, "ERROR allocating matrix C: %s\n", cudaGetErrorString(err));

    // Copy data from host PC to GPU
    err = cudaMemcpy(dev_matrixA, host_matrixA, matrixA_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying matrix A to GPU: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(dev_matrixB, host_matrixB, matrixB_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying matrix B to GPU: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(dev_matrixC, host_matrixC, matrixC_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying matrix C to GPU: %s\n", cudaGetErrorString(err));
}

//==================================================================================
//=== CUDA Vector Kernels
//==================================================================================
/**
 * @required ALL VECTORS MUST BE THE SAME LENGTH
 * @brief - kernel for GPU computation of a vector addition
 * @param dev_vecA - pointer to device memory for vector A
 * @param dev_vecB - pointer to device memory for vector B
 * @param dev_vecC - pointer to device memory for vector C
 * @param alpha - multiplier for values in vector A
 * @param beta - multiplier for values in vector B
 * @param vecLen - length of all vectors
 */
__global__ void VectorAdditionKernel(float *dev_vecA, float *dev_vecB, float *dev_vecC,
                                     float alpha, float beta, int vecLen) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vecLen) {
        dev_vecC[i] = alpha * dev_vecA[i] + beta * dev_vecB[i];
    }
}

/**
 * @required ALL VECTORS MUST BE THE SAME LENGTH
 * @brief - kernel for GPU computation of a vector hadamard product
 * @param dev_vecA - pointer to device memory for vector A
 * @param dev_vecB - pointer to device memory for vector B
 * @param dev_vecC - pointer to device memory for vector C
 * @param alpha - multiplier for values in vector A
 * @param beta - multiplier for values in vector B
 * @param vecLen - length of all vectors
 */
__global__ void VectorHadamardKernel(float *dev_vecA, float *dev_vecB, float *dev_vecC,
                                     float alpha, float beta, int vecLen) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vecLen) {
        dev_vecC[i] = alpha * dev_vecA[i] * beta * dev_vecB[i];
    }
}

/**
 * @required ALL VECTORS MUST BE THE SAME LENGTH
 *           REMEMBER: Call kernel using: <<<grid, threads, vecLen>>>
 * @brief - kernel for GPU computation of a vector dot product
 * @param dev_vecA - pointer to device memory for vector A
 * @param dev_vecB - pointer to device memory for vector B
 * @param result - pointer to a single float value where the result will be returned
 * @param alpha - multiplier for values in vector A
 * @param beta - multiplier for values in vector B
 * @param vecLen - length of all vectors
 */
__global__ void VectorDotProduct(float *dev_vecA, float *dev_vecB, float *result,
                                 float alpha, float beta, int vecLen) {
    extern __shared__ float temp[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vecLen) {
        temp[i] = alpha * dev_vecA[i] * beta * dev_vecB[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0;
        for (int j = 0; j < vecLen; j++) {
            sum += temp[j];
        }
        result[0] = sum;
    }
}

/**
 * @required INPUT AND OUTPUT VECTORS MUST BE THE SAME LENGTH
 * @brief - kernel for GPU computation of the vector sigmoid function
 * @param dev_matrixA - pointer to device memory for vector A
 * @param dev_matrixC - pointer to device memory for vector C
 * @param vecLen - length of all vectors
 */
__global__ void VectorSigmoid(float *dev_vecA, float *dev_vecC, int vecLen) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < vecLen) {
        float exp = 1 + expf(-dev_vecA[index]);
        dev_vecC[index] = 1 / exp;
    }
}

/**
 * @required INPUT AND OUTPUT VECTORS MUST BE THE SAME LENGTH
 * @brief - kernel for GPU computation of the vector sigmoid derivative function
 * @param dev_matrixA - pointer to device memory for vector A
 * @param dev_matrixC - pointer to device memory for vector C
 * @param vecLen - length of all vectors
 */
__global__ void VectorSigmoidDerivative(float *dev_vecA, float *dev_vecC, int vecLen) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < vecLen) {
        float exp = 1 + expf(-dev_vecA[index]);
        float sig = 1 / exp;
        dev_vecC[index] = sig * (1 - sig);
    }
}

//==================================================================================
//=== CUDA Vector Kernel Drivers
//==================================================================================

/**
 * @brief driver function for computing vector operations
 * @param argc - from compiler
 * @param argv - from compiler
 * @param devID - device ID number
 * @param vectorSize - reference to vector size structure
 * @param operation - switch-case value for which matrix operation to perform
 *                    1: Vector addition
 *                    2: Vector Hadamard product
 *                    3: Vector dot product
 *                    4: Vector sigmoid function
 *                    5: Vector sigmoid derivative
 * @param host_vectorA - pointer to host vector A (with values)
 * @param host_vectorB - pointer to host vector B (with values)
 * @param host_vectorC - pointer to host vector C (with values)
 * @param alpha - multiplier for values in vector A
 * @param beta - multiplier for values in vector B
 */
void RunVectorKernel(int argc, char **argv, int &devID, VectorSize *vectorSize, int operation,
                     float *host_vectorA, float *host_vectorB, float *host_vectorC, float alpha, float beta) {
    // Assign CUDA variables
    cudaError_t err;
    dim3 threads(NUM_THREADS, NUM_THREADS);
    int gridX = (int) ceil((float) vectorSize->len_C / (float) threads.x);
    int gridY = (int) ceil((float) vectorSize->len_C / (float) threads.y);
    dim3 grid((unsigned int) gridX, (unsigned int) gridY);

    // Assign computation variables
    float *dev_vectorA = NULL;
    float *dev_vectorB = NULL;
    float *dev_vectorC = NULL;

    size_t vectorC_size = vectorSize->len_C * sizeof(float);

    // Initialize memory on GPU
    VectorInitCUDA(argc, argv, devID, vectorSize, host_vectorA, host_vectorB, dev_vectorA, dev_vectorB, dev_vectorC);

    switch (operation) {
        case 1: {
            // Compute vector addition
            VectorAdditionKernel<<<grid, threads>>>(dev_vectorA, dev_vectorB, dev_vectorC, alpha, beta,
                    vectorSize->len_C);
            err = cudaGetLastError();
            if (err != cudaSuccess) fprintf(stderr, "ERROR in Vector Add Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 2: {
            // Compute vector Hadamard Product
            VectorHadamardKernel<<<grid, threads>>>(dev_vectorA, dev_vectorB, dev_vectorC, alpha, beta,
                    vectorSize->len_C);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Vector Hadamard Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 3: {
            // Compute vector dot product
            VectorDotProduct<<<grid, threads, vectorSize->len_C>>>
                                              (dev_vectorA, dev_vectorB, dev_vectorC, alpha, beta, vectorSize->len_C);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Vector Dot product Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 4: {
            // Compute sigmoid function
            VectorSigmoid<<<grid, threads>>>(dev_vectorA, dev_vectorC, vectorSize->len_C);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Vector Sigmoid Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 5: {
            // Compute sigmoid derivative
            VectorSigmoidDerivative<<<grid, threads>>>(dev_vectorA, dev_vectorC, vectorSize->len_C);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Vector Sigmoid Derivative Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        default: {
            fprintf(stderr, "ERROR: No vector kernel selected. Operation Aborted");
            break;
        }
    }

    // Make sure device is finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        fprintf(stderr, "ERROR synchronizing Vector Kernel calculation: %s\n", cudaGetErrorString(err));

    // Copy data from GPU to host PC
    err = cudaMemcpy(host_vectorC, dev_vectorC, vectorC_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        fprintf(stderr, "ERROR copying vector C to Host: %s\n", cudaGetErrorString(err));

    // Free GPU memory
    err = cudaFree(dev_vectorA);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing vector A on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_vectorB);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing vector B on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_vectorC);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing vector C on GPU: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "ERROR synchronizing Vector Kernel End: %s\n", cudaGetErrorString(err));
    if (LOGGING == 1) fprintf(stdout, "Vector Kernel finished.\n");
}

//==================================================================================
//=== CUDA Matrix Kernels
//==================================================================================

/**
 * @required ALL MATRICES MUST BE THE SAME DIMENSIONS
 * @brief - kernel for GPU computation of matrix additions
 * @param dev_matrixA - pointer to device memory for matrix A
 * @param dev_matrixB - pointer to device memory for matrix B
 * @param dev_matrixC - pointer to device memory for matrix C
 * @param alpha - multiplier for values in matrix A
 * @param beta - multiplier for values in matrix B
 * @param matrix_width - width of all matrices
 * @param matrix_height - height of all matrices
 */
__global__ void MatrixAddKernel(float *dev_matrixA, float *dev_matrixB, float *dev_matrixC,
                                float alpha, float beta, int matrix_width, int matrix_height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + row * matrix_height;
    if (col < matrix_width && row < matrix_height) {
        dev_matrixC[index] = alpha * dev_matrixA[index] + beta * dev_matrixB[index];
    }
}

/**
 * @required ALL MATRICES MUST BE THE SAME DIMENSIONS
 * @brief - kernel for actual GPU computation for the matrix Hadamard product
 * @param dev_matrixA - pointer to device memory for matrix A
 * @param dev_matrixB - pointer to device memory for matrix B
 * @param dev_matrixC - pointer to device memory for matrix C
 * @param alpha - multiplier for values in matrix A
 * @param beta - multiplier for values in matrix B
 * @param matrix_width - width of all matrices
 * @param matrix_height - height of all matrices
 */
__global__ void MatrixHadamardKernel(float *dev_matrixA, float *dev_matrixB, float *dev_matrixC,
                                     float alpha, float beta, int matrix_width, int matrix_height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + row * matrix_height;
    if (col < matrix_width && row < matrix_height) {
        dev_matrixC[index] = alpha * dev_matrixA[index] * beta * dev_matrixB[index];
    }
}

/**
 * @required ALL MATRICES MUST BE THE SAME DIMENSIONS
 * @brief - kernel for GPU computation of matrix sigmoid function
 * @param dev_matrixA - pointer to device memory for matrix A
 * @param dev_matrixC - pointer to device memory for matrix C
 * @param matrix_width - width of all matrices
 * @param matrix_height - height of all matrices
 */
__global__ void MatrixSigmoid(float *dev_matrixA, float *dev_matrixC,
                              int matrix_width, int matrix_height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + row * matrix_height;
    if (col < matrix_width && row < matrix_height) {
        float exp = 1 + expf(-dev_matrixA[index]);
        dev_matrixC[index] = 1 / exp;
    }
}

/**
 * @required ALL MATRICES MUST BE THE SAME DIMENSIONS
 * @brief - kernel for GPU computation of the matrix sigmoid derivative function
 * @param dev_matrixA - pointer to device memory for matrix A
 * @param dev_matrixC - pointer to device memory for matrix C
 * @param matrix_width - width of all matrices
 * @param matrix_height - height of all matrices
 */
__global__ void MatrixSigmoidDerivative(float *dev_matrixA, float *dev_matrixC,
                                        int matrix_width, int matrix_height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + row * matrix_height;
    if (col < matrix_width && row < matrix_height) {
        float exp = 1 + expf(-dev_matrixA[index]);
        float sig = 1 / exp;
        dev_matrixC[index] = sig * (1 - sig);
    }
}

//==================================================================================
//=== CUDA Matrix Kernel Drivers
//==================================================================================

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

void MatrixMultiplyCUBLAS(int argc, char **argv, int &devID, MatrixSize *matrixSize,
                          float *host_matrixA, float *host_matrixB, float *host_matrixC,
                          float alpha, float beta, bool transposeA, bool transposeB) {
    // Assign CUDA variables
    cublasHandle_t handle;
    cudaError_t err;
    cublasCreate(&handle);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devID);
    dim3 threads(NUM_THREADS, NUM_THREADS);
    dim3 grid(matrixSize->C_width / threads.x, matrixSize->C_height / threads.y);

    // Assign computation variables
    float *dev_matrixA = NULL, *dev_matrixB = NULL, *dev_matrixC = NULL;
    int m = matrixSize->A_height;
    int n = matrixSize->B_width;
    int k = matrixSize->A_width;
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    if (transposeA) transA = CUBLAS_OP_T;
    if (transposeB) transB = CUBLAS_OP_T;
    size_t matrixC_size = matrixSize->C_height * matrixSize->C_width * sizeof(float);

    // Initialize memory on GPU
    MatrixInitCUDA(argc, argv, devID, matrixSize,
                   host_matrixA, host_matrixB, host_matrixC,
                   dev_matrixA, dev_matrixB, dev_matrixC);

    // Perform matrix multiplication
    // SGEMM PARAMS: (handle, transposeA, transposeB, m, n, k, alpha, matrix A, k, matrix B, n, beta, matrix C, n)
    cublasSgemm(handle, transA, transB, m, n, k, &alpha, dev_matrixA, m,
                dev_matrixB, n, &beta, dev_matrixC, m);
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "ERROR in SGEMM: %s\n", cudaGetErrorString(err));

    // Make sure device is finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "ERROR synchronizing SGEMM calculation: %s\n", cudaGetErrorString(err));

    // Copy data from GPU to host PC
    err = cudaMemcpy(host_matrixC, dev_matrixC, matrixC_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying matrix C to Host: %s\n", cudaGetErrorString(err));

    // Free GPU memory
    err = cudaFree(dev_matrixA);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing matrix A on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixB);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing matrix B on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixC);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing matrix C on GPU: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "ERROR synchronizing SGEMM end: %s\n", cudaGetErrorString(err));
    if (LOGGING == 1) fprintf(stdout, "Matrix Kernel finished.\n");
}

/**
 * @required ALL MATRICES MUST BE THE SAME DIMENSIONS
 * @brief driver function for computing the matrix operations
 * @param argc - from compiler
 * @param argv - from compiler
 * @param devID - device ID number
 * @param matrixSize - reference to matrix size structure
 * @param operation - switch-case value for which matrix operation to perform
 *                    1: Matrix addition
 *                    2: Matrix Hadamard product
 *                    3: Sigmoid function
 *                    4: Sigmoid derivative
 * @param host_matrixA - pointer to host matrix A (with values)
 * @param host_matrixB - pointer to host matrix B (with values)
 * @param host_matrixC - pointer to host matrix C (with values)
 * @param alpha - multiplier for values in matrix A
 * @param beta - multiplier for values in matrix B
 */
void RunMatrixKernel(int argc, char **argv, int &devID, MatrixSize *matrixSize, int operation,
                     float *host_matrixA, float *host_matrixB, float *host_matrixC, float alpha, float beta) {
    // Assign CUDA variables
    cudaError_t err;
    dim3 threads(NUM_THREADS, NUM_THREADS);
    int gridX = (int) ceil((float) matrixSize->C_width / (float) threads.x);
    int gridY = (int) ceil((float) matrixSize->C_height / (float) threads.y);
    dim3 grid((unsigned int) gridX, (unsigned int) gridY);

    // Assign computation variables
    float *dev_matrixA = NULL, *dev_matrixB = NULL, *dev_matrixC = NULL;
    size_t matrixC_size = matrixSize->C_height * matrixSize->C_width * sizeof(float);

    // Initialize memory on GPU
    MatrixInitCUDA(argc, argv, devID, matrixSize,
                   host_matrixA, host_matrixB, host_matrixC,
                   dev_matrixA, dev_matrixB, dev_matrixC);

    switch (operation) {
        case 1: {
            // Compute Matrix Addition
            MatrixAddKernel<<<grid, threads>>>(dev_matrixA, dev_matrixB, dev_matrixC, alpha, beta,
                    matrixSize->C_width, matrixSize->C_height);
            err = cudaGetLastError();
            if (err != cudaSuccess) fprintf(stderr, "ERROR in Matrix Add Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 2: {
            // Compute Hadamard Product
            MatrixHadamardKernel<<<grid, threads>>>(dev_matrixA, dev_matrixB, dev_matrixC, alpha, beta,
                    matrixSize->C_width, matrixSize->C_height);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Matrix Hadamard Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 3: {
            // Compute Sigmoid function
            MatrixSigmoid<<<grid, threads>>>(dev_matrixA, dev_matrixC, matrixSize->C_width, matrixSize->C_height);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Matrix Sigmoid Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        case 4: {
            // Compute Sigmoid derivative function
            MatrixSigmoidDerivative<<<grid, threads>>>
                                            (dev_matrixA, dev_matrixC, matrixSize->C_width, matrixSize->C_height);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "ERROR in Matrix Sigmoid Derivative Computation: %s\n", cudaGetErrorString(err));
            break;
        }
        default: {
            fprintf(stderr, "ERROR: No matrix kernel selected. Operation Aborted");
            break;
        }
    }

    // Make sure device is finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        fprintf(stderr, "ERROR synchronizing Matrix Kernel calculation: %s\n", cudaGetErrorString(err));

    // Copy data from GPU to host PC
    err = cudaMemcpy(host_matrixC, dev_matrixC, matrixC_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "ERROR copying matrix C to Host: %s\n", cudaGetErrorString(err));

    // Free GPU memory
    err = cudaFree(dev_matrixA);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing matrix A on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixB);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing matrix B on GPU: %s\n", cudaGetErrorString(err));
    err = cudaFree(dev_matrixC);
    if (err != cudaSuccess) fprintf(stderr, "ERROR freeing matrix C on GPU: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "ERROR synchronizing Matrix Kernel end: %s\n", cudaGetErrorString(err));
    if (LOGGING == 1) fprintf(stdout, "Matrix Kernel finished.\n");

}

//==================================================================================
//=== Test Function
//==================================================================================

void runTest(int argc, char **argv, int devID) {
    int N = 10;
    float *host_A, *host_B, *host_C, *host_D;
    float *host_vA, *host_vB, *host_vC, *host_vD, *host_vE;

    // Create matrices
    MatrixSize *testMatrixSize = (MatrixSize *) calloc(sizeof(MatrixSize), 1);
    size_t calcSize = N * N * sizeof(float);
    host_A = (float *) calloc(calcSize, 1);
    host_B = (float *) calloc(calcSize, 1);
    host_C = (float *) calloc(calcSize, 1);
    host_D = (float *) calloc(calcSize, 1);
    SetMatrixSize(testMatrixSize, N, N, N, N, N + 1, N + 1);

    // Create vectors
    VectorSize *testVectorSize = (VectorSize *) calloc(sizeof(VectorSize), 1);
    size_t calcSize_V = N * sizeof(float);
    host_vA = (float *) calloc(calcSize_V, 1);
    host_vB = (float *) calloc(calcSize_V, 1);
    host_vC = (float *) calloc(calcSize_V, 1);
    host_vD = (float *) calloc(calcSize_V, 1);
    host_vE = (float *) calloc(calcSize_V, 1);
    SetVectorSize(testVectorSize, N);

    // Initialize matrix values
    for (int i = 0; i < N * N; i++) {
        host_A[i] = (float) i;
        host_B[i] = (float) i;
    }

    // Initialize vector values
    for (int i = 0; i < N; i++) {
        host_vA[i] = (float) i;
        host_vB[i] = (float) i;
    }

    // MATRIX TESTS

    if (LOGGING == 1) {
        fprintf(stdout, "Matrix A:\n");
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                fprintf(stdout, "%6.0f ", host_A[i * j]);
            }
            fprintf(stdout, "\n");
        }

        fprintf(stdout, "\nMatrix B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(stdout, "%6.0f ", host_B[i * j]);
            }
            fprintf(stdout, "\n");
        }
    }

    RunMatrixKernel(argc, argv, devID, testMatrixSize, 3, host_A, host_B, host_C, 1.0, 1.0);
    RunMatrixKernel(argc, argv, devID, testMatrixSize, 4, host_A, host_B, host_D, 1.0, 1.0);

    if (LOGGING == 1) {
        fprintf(stdout, "\nMatrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(stdout, "%6.10f ", host_C[i * j]);
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\nMatrix D:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(stdout, "%6.10f ", host_D[i * j]);
            }
            fprintf(stdout, "\n");
        }
    }

    // VECTOR TESTS

    if (LOGGING == 1) {
        fprintf(stdout, "Vector A:\n");
        for (int i = 0; i < N; i++) {
            fprintf(stdout, "%6.0f ", host_vA[i]);
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "\nVector B:\n");

        for (int i = 0; i < N; i++) {
            fprintf(stdout, "%6.0f ", host_vB[i]);
        }
        fprintf(stdout, "\n");
    }

    RunVectorKernel(argc, argv, devID, testVectorSize, 3, host_vA, host_vB, host_vC, 1.0, 1.0);
    RunVectorKernel(argc, argv, devID, testVectorSize, 4, host_vA, host_vB, host_vD, 1.0, 1.0);
    RunVectorKernel(argc, argv, devID, testVectorSize, 5, host_vA, host_vB, host_vE, 1.0, 1.0);

    if (LOGGING == 1) {
        fprintf(stdout, "Vector C:\n");
        for (int i = 0; i < N; i++) {
            fprintf(stdout, "%6.0f ", host_vC[i]);
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "\nVector D:\n");

        for (int i = 0; i < N; i++) {
            fprintf(stdout, "%6.10f ", host_vD[i]);
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "\nVector E:\n");

        for (int i = 0; i < N; i++) {
            fprintf(stdout, "%6.10f ", host_vE[i]);
        }
        fprintf(stdout, "\n");
    }
}

//==================================================================================
//=== Utility Functions
//==================================================================================

void ReadCSV(std::ifstream &file, int elements, float *array)
{
    std::string csvData;
    getline(file, csvData);

    std::istringstream dataStream(csvData);

    for (int col = 0; col < elements; col++){
        std::string value;
        getline(dataStream, value, ',');
        if ( !dataStream.good() )
            break;
        std::istringstream convertor(value);
        convertor >> array[col];
    }
}

void InitializeWeights(float *weights, MatrixSize *dims)
{
    int cols = dims->C_width;
    int rows = dims->C_height;
    int numEl = cols*rows;
    for(int idx = 0; idx < numEl; idx++)
    {
        weights[idx] = ((float) rand() / (RAND_MAX));
    }
}

//==================================================================================
//=== Main Function
//==================================================================================

/**
 * @brief computes weight matrices for a shallow neural network
 * @param argc - from compiler
 * @param argv - from compiler
 * @return 0 if success
 */
int main(int argc, char **argv) {
    // Assign CUDA variables
    int devID = 0;
    cudaGetDevice(&devID);
    cudaError_t mainErr;
    //runTest(argc, argv, devID);

    // Define NN layer lengths
    unsigned int layer_1 = 784;
    unsigned int layer_2 = 128;
    unsigned int layer_3 = 10;

    // Allocate memory for matrices and vectors
    float *a1, *a2, *a3;    // Activation vectors
    float *z2, *z3;         // Pre-sigmoid intermediary vectors
    float *W1, *W2;         // Weight matrices
    float *y;               // One-hot result vector
    float *del3, *del2;     // Error vectors
    float *scratch1, *scratch2;     // Error vectors
    float *Del2, *Del1;     // Error gradients

    a1 = (float *) calloc((size_t) layer_1, sizeof(float));
    a2 = (float *) calloc((size_t) layer_2, sizeof(float));
    a3 = (float *) calloc((size_t) layer_3, sizeof(float));
    z2 = (float *) calloc((size_t) layer_2, sizeof(float));
    z3 = (float *) calloc((size_t) layer_3, sizeof(float));
    y = (float *) calloc((size_t) layer_3, sizeof(float));
    W1 = (float *) calloc((size_t) layer_2 * layer_1, sizeof(float));
    W2 = (float *) calloc((size_t) layer_3 * layer_2, sizeof(float));
    del3 = (float *) calloc((size_t) layer_3, sizeof(float));
    del2 = (float *) calloc((size_t) layer_2, sizeof(float));
    scratch1 = (float *) calloc((size_t) layer_2, sizeof(float));
    scratch2 = (float *) calloc((size_t) layer_2, sizeof(float));
    Del2 = (float *) calloc((size_t) layer_3 * layer_2, sizeof(float));
    Del1 = (float *) calloc((size_t) layer_2 * layer_1, sizeof(float));

    // Initialize vector and matrix size structures for computation
    MatrixSize *inter2 = (MatrixSize *) calloc(sizeof(MatrixSize), 1);
    MatrixSize *inter3 = (MatrixSize *) calloc(sizeof(MatrixSize), 1);
    MatrixSize *grad1 = (MatrixSize *) calloc(sizeof(MatrixSize), 1);
    MatrixSize *grad2 = (MatrixSize *) calloc(sizeof(MatrixSize), 1);
    MatrixSize *backprop1 = (MatrixSize *) calloc(sizeof(MatrixSize), 1);
    MatrixSize *backprop2 = (MatrixSize *) calloc(sizeof(MatrixSize), 1);

    VectorSize *activation2 = (VectorSize *) calloc(sizeof(VectorSize), 1);
    VectorSize *activation3 = (VectorSize *) calloc(sizeof(VectorSize), 1);
    VectorSize *delta2 = (VectorSize *) calloc(sizeof(VectorSize), 1);
    VectorSize *delta3 = (VectorSize *) calloc(sizeof(VectorSize), 1);

    if (LOGGING == 1) fprintf(stdout, "Intermediate 2: ");
    SetMatrixSize(inter2, 1, layer_1, layer_2, layer_1, 1, layer_2);
    if (LOGGING == 1) fprintf(stdout, "Intermediate 3: ");
    SetMatrixSize(inter3, 1, layer_2, layer_3, layer_2, 1, layer_3);
    if (LOGGING == 1) fprintf(stdout, "Grad 1: ");
    SetMatrixSize(grad1, 1, layer_2, 1, layer_1, layer_2, layer_1);
    if (LOGGING == 1) fprintf(stdout, "Grad 2: ");
    SetMatrixSize(grad2, 1, layer_3, 1, layer_2, layer_3, layer_2);
    if (LOGGING == 1) fprintf(stdout, "Backprop 1: ");
    SetMatrixSize(backprop1, layer_2, layer_1, layer_2, layer_1, layer_2, layer_1);
    if (LOGGING == 1) fprintf(stdout, "Backprop 2: ");
    SetMatrixSize(backprop2, layer_3, layer_2, layer_3, layer_2, layer_3, layer_2);

    if (LOGGING == 1) fprintf(stdout, "Activation 2: ");
    SetVectorSize(activation2, layer_2);
    if (LOGGING == 1) fprintf(stdout, "Activation 3: ");
    SetVectorSize(activation3, layer_3);
    if (LOGGING == 1) fprintf(stdout, "Delta 2: ");
    SetVectorSize(delta2, layer_2);
    if (LOGGING == 1) fprintf(stdout, "Delta 3: ");
    SetVectorSize(delta3, layer_3);

    // Set number of epochs and samples
    int epochs = 1; // Number of training epochs (iterations through data)
    int num_train = 20000;   // Number of samples;
    int num_test = 5000;

    // Initalize weights
    InitializeWeights(W1, grad1);
    InitializeWeights(W2, grad2);

    //Perform neural network training
    for (int epoch = 0; epoch < epochs; epoch++) {

        // Open training data files
        std::ifstream x_train_data("./data/train_img.csv");
        std::ifstream y_train_data("./data/train_res.csv");

        for (int sample = 0; sample < num_train; sample++) {
            // LOAD a1 AND y VECTORS:
            ReadCSV(x_train_data, layer_1, a1);
            ReadCSV(y_train_data, layer_3, y);

            // FORWARD PROPOGATION:
            MatrixMultiplyCUBLAS(argc, argv, devID, inter2, a1, W1, z2, 1.0, 1.0, false, true); // Compute z2
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "z2 Computation: %s\n", cudaGetErrorString(mainErr));
            RunVectorKernel(argc, argv, devID, activation2, 4, z2, z2, a2, 1.0, 1.0);           // Compute a2
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "a2 Computation: %s\n", cudaGetErrorString(mainErr));
            MatrixMultiplyCUBLAS(argc, argv, devID, inter3, a2, W2, z3, 1.0, 1.0, false, true); // Compute z3
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "z3 Computation: %s\n", cudaGetErrorString(mainErr));
            RunVectorKernel(argc, argv, devID, activation3, 4, z3, z3, a3, 1.0, 1.0);           // Compute a3
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "a3 Computation: %s\n", cudaGetErrorString(mainErr));

            // BACKWARD PROPOGATION:
            RunVectorKernel(argc, argv, devID, delta3, 1, z3, y, del3, 1.0, (float) -1.0);           // Compute del3
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "del3 Computation: %s\n", cudaGetErrorString(mainErr));

            MatrixMultiplyCUBLAS(argc, argv, devID, inter3, del3, W2, scratch1, 1.0, 1.0, false, false); // Compute del2 lhs
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "pre-del2 lhs Computation: %s\n", cudaGetErrorString(mainErr));

            RunVectorKernel(argc, argv, devID, delta2, 5, z2, y, scratch2, 1.0, (float) -1.0);           // Compute del2 rhs
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "pre-del2 rhs Computation: %s\n", cudaGetErrorString(mainErr));

            RunVectorKernel(argc, argv, devID, delta2, 2, scratch1, scratch2, del2, 1.0, (float) -1.0);  // Compute del2
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "del2 Computation: %s\n", cudaGetErrorString(mainErr));

            MatrixMultiplyCUBLAS(argc, argv, devID, grad1, del2, a1, Del1, 1.0, 1.0, true, false);   // Compute Del1
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "Del1 Computation: %s\n", cudaGetErrorString(mainErr));

            MatrixMultiplyCUBLAS(argc, argv, devID, grad2, del3, a2, Del2, 1.0, 1.0, true, false);   // Compute Del2
            mainErr = cudaGetLastError();
            if (mainErr != cudaSuccess) fprintf(stderr, "Del2 Computation: %s\n", cudaGetErrorString(mainErr));

            // Gradient descent
            RunMatrixKernel(argc, argv, devID, backprop1, 1, W1, Del1, W1, 1.0,
                            (float) -1.0 / (float) num_train); // Compute new W1
            RunMatrixKernel(argc, argv, devID, backprop2, 1, W2, Del2, W2, 1.0,
                            (float) -1.0 / (float) num_train); // Compute new W2
            cudaDeviceSynchronize();
            if( (sample % 1000) == 0) printf("Iteration: %d\n", sample);
            if( (sample % 5000) == 0) cudaDeviceReset();
        }
        // Close training data files
        x_train_data.close();
        y_train_data.close();

    }

    // Open verification data files
    std::ifstream x_test_data("./data/tests_img.csv");
    std::ifstream y_test_data("./data/tests_res.csv");

    int correct = 0;

    for(int test_sample = 0; test_sample < num_test; test_sample++)
    {
        // LOAD a1 AND y VECTORS:
        ReadCSV(x_test_data, layer_1, a1);
        ReadCSV(y_test_data, layer_3, y);

        // FORWARD PROPOGATION:
        MatrixMultiplyCUBLAS(argc, argv, devID, inter2, a1, W1, z2, 1.0, 1.0, false, true); // Compute z2
        mainErr = cudaGetLastError();
        if (mainErr != cudaSuccess) fprintf(stderr, "z2 Computation: %s\n", cudaGetErrorString(mainErr));
        RunVectorKernel(argc, argv, devID, activation2, 4, z2, z2, a2, 1.0, 1.0);           // Compute a2
        mainErr = cudaGetLastError();
        if (mainErr != cudaSuccess) fprintf(stderr, "a2 Computation: %s\n", cudaGetErrorString(mainErr));
        MatrixMultiplyCUBLAS(argc, argv, devID, inter3, a2, W2, z3, 1.0, 1.0, false, true); // Compute z3
        mainErr = cudaGetLastError();
        if (mainErr != cudaSuccess) fprintf(stderr, "z3 Computation: %s\n", cudaGetErrorString(mainErr));
        RunVectorKernel(argc, argv, devID, activation3, 4, z3, z3, a3, 1.0, 1.0);           // Compute a3
        mainErr = cudaGetLastError();
        if (mainErr != cudaSuccess) fprintf(stderr, "a3 Computation: %s\n", cudaGetErrorString(mainErr));

        float a3max = 0.0;
        int a3max_idx = 0;
        int ymax_idx = 0;

        for(int i = 0; i < layer_3; i++)
        {
            if(a3[i] > a3max)
            {
                a3max = a3[i];
                a3max_idx = i;
            }
            if(y[i] == 1) ymax_idx = i;
        }
        if(ymax_idx == a3max_idx) correct++;
    }
    x_test_data.close();
    y_test_data.close();
    printf("The network correctly identified %d of %d samples\n", correct, num_test);
    return 0;
}
