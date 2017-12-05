#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

__global__ void prescan(double *input, double *output, double *scratch, int N)
{
    extern __shared__ double temp[];

    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    int blockThread = threadIdx.x;
    int arrayIndex = 2 * blockThread;

    int offset = 1;

    temp[arrayIndex] = input[2*thread];
    temp[arrayIndex + 1] = input[2*thread + 1];

    for (int d = blockDim.x; d > 0; d = d / 2)
    {
        __syncthreads();

        if (blockThread < d)
        {
            int ai = offset * (arrayIndex + 1) - 1;
            int bi = offset * (arrayIndex + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (blockThread == 0)
    {
        if (scratch) scratch[blockIdx.x] = temp[N - 1];
        temp[N - 1] = 0;
    }

    for (int d = 1; d < blockDim.x * 2; d *= 2)
    {
        offset = offset / 2;
        __syncthreads();

        if (blockThread < d)
        {
            int ai = offset * (arrayIndex + 1) - 1;
            int bi = offset * (arrayIndex + 2) - 1;
            double t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[2*thread] = temp[arrayIndex];
    output[2*thread + 1] = temp[arrayIndex + 1];
}

__global__ void prescanSum (double *to_add, double *result, int N)
{
    double addition = to_add[blockIdx.x];
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    result[thread] += addition;
}

void scanCPU(double *f_out, double *f_in, int i_n)
{
    f_out[0] = 0;
    for (int i = 0; i < i_n -1; i++)
    {
        f_out[i+1] = f_out[i] + f_in[i];
    }

}

double myDiffTime(struct timeval &start, struct timeval &end)
{
    double d_start, d_end;
    d_start = (double) (start.tv_sec + start.tv_usec / 1000000.0);
    d_end = (double) (end.tv_sec + end.tv_usec / 1000000.0);
    return (d_end - d_start);
}

int main(int argc, char **argv)
{
    // Running parameters
    int numThreads = 1024;
    int N = atoi(argv[1]);

    // Init CUDA size variables
    int numBlocks = N / numThreads;
    if (N % numThreads != 0) numBlocks++;
    int vecLen = numBlocks * numThreads;
    const dim3 blockSize(numThreads / 2, 1, 1);
    const dim3 gridSize(numBlocks, 1, 1);

//    printf("Number of Blocks: %d\nNumber of Threads: %d\nVecLen: %d\nN: %d\n", numBlocks, numThreads, vecLen, N);
//    printf("Blocksize: %d\nGridsize: %d\n",blockSize.x, gridSize.x);

    // Host Memory Allocation
    double *host_CPU, *host_input, *host_GPU, *host_scratch, *host_addition;

    host_CPU = (double *) calloc(vecLen, sizeof(double));
    host_input = (double *) calloc(vecLen, sizeof(double));
    host_GPU = (double *) calloc(vecLen, sizeof(double));
    host_scratch = (double *) calloc(vecLen, sizeof(double));
    host_addition = (double *) calloc(vecLen, sizeof(double));


    double d_gpuFullTime, d_gpuScanTime, d_cpuTime;
    timeval fullStart, fullEnd, scanStart, scanEnd;

    // Device Memory Allocation
    double *dev_GPU, *dev_input, *dev_scratch, *dev_addition;
    cudaError_t err;
    err = cudaMalloc((void **) &dev_input, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Malloc of dev_input: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_GPU, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Malloc of dev_GPU: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_scratch, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Malloc of dev_scratch: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &dev_addition, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Malloc of dev_addition: %s\n", cudaGetErrorString(err));

    err = cudaMemset(dev_input, 0, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memset of dev_input: %s\n", cudaGetErrorString(err));
    err = cudaMemset(dev_GPU, 0, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memset of dev_GPU: %s\n", cudaGetErrorString(err));
    err = cudaMemset(dev_scratch, 0, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memset of dev_scratch: %s\n", cudaGetErrorString(err));
    err = cudaMemset(dev_addition, 0, vecLen * sizeof(double));
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memset of dev_addition: %s\n", cudaGetErrorString(err));


    // Generate array values
    for (int i = 0; i < N; i++)
    {
//        host_In[i] = (double) (rand() % 1000000) / 1000.0;
        host_input[i] = (double) i * 1.0;
    }

    gettimeofday(&fullStart, NULL);
    // Copy from host to device
    cudaMemcpy(dev_input, host_input, vecLen * sizeof(double), cudaMemcpyHostToDevice);

    // Perform scan on CUDA
    // START SCAN AND GET SCAN START TIME
    gettimeofday(&scanStart, NULL);

    prescan<<<gridSize, blockSize, numThreads * sizeof(double)>>>(dev_input, dev_GPU, dev_scratch, numThreads);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "ERROR in prescan: %s\n", cudaGetErrorString(err));

    // ACCUMULATE ADDITIONS
    prescan<<<dim3(1,1,1), blockSize, numThreads * sizeof(double) >>>(dev_scratch, dev_addition, NULL, numThreads);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "ERROR in creating addition vector: %s\n", cudaGetErrorString(err));

    // ADD TO ELEMENTS FOR TRUE SUM
    prescanSum<<<gridSize, dim3(numThreads,1,1)>>>(dev_addition,dev_GPU, N);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "ERROR in addition: %s\n", cudaGetErrorString(err));

    // GET SCAN END TIME
    gettimeofday(&scanEnd, NULL);

    // Copy memory back and get full end time
    err = cudaMemcpy(host_GPU, dev_GPU, vecLen*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memcpy back to host: %s\n", cudaGetErrorString(err));

    gettimeofday(&fullEnd, NULL);


    err = cudaMemcpy(host_scratch, dev_scratch, vecLen*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memcpy back to host: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(host_addition, dev_addition, vecLen*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in memcpy back to host: %s\n", cudaGetErrorString(err));

    d_gpuFullTime = myDiffTime(fullStart, fullEnd);
    d_gpuScanTime = myDiffTime(scanStart, scanEnd);

    // Perform serial scan
    gettimeofday(&fullStart, NULL);
    scanCPU(host_CPU, host_input, N);
    gettimeofday(&fullEnd, NULL);
    d_cpuTime = myDiffTime(fullStart, fullEnd);


    for (int i = 0; i < N; i++)
    {
        if (host_CPU[i] - host_GPU[i] > 0.1)
        {
        	printf("Results do not match! c[%i]=%f, g[%i]=%f\n", i, host_CPU[i], i, host_GPU[i]);
            break;
        }
    }

    printf("%i\t%f\t%f\t%f\n", N, d_cpuTime,d_gpuScanTime,d_gpuFullTime);
}