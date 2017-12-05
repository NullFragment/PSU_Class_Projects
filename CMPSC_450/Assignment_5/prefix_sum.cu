#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <curand_mtgp32_kernel.h>

#define NUM_THREADS 1024

__global__ void prescan(double *g_odata, double *g_idata, int n);

void scanCPU(double *f_out, double *f_in, int i_n);

double myDiffTime(struct timeval &start, struct timeval &end)
{
    double d_start, d_end;
    d_start = (double) (start.tv_sec + start.tv_usec / 1000000.0);
    d_end = (double) (end.tv_sec + end.tv_usec / 1000000.0);
    return (d_end - d_start);
}

int main(int argc, char **argv)
{
    // Get array size from input args
    int N = atoi(argv[1]);

    // Create host variables
    double
            *host_A = (double * ) calloc(N, sizeof(double)),
            *host_C = (double * ) calloc(N, sizeof(double)),
            *host_G = (double * ) calloc(N, sizeof(double)),
            d_gpuFullTime,
            d_gpuScanTime,
            d_cpuTime;
    int vecSize = N * sizeof(double);
    timeval fullStart, fullEnd, scanStart, scanEnd;
    int numBlocks = N/NUM_THREADS;
    dim3 blockShape(NUM_THREADS/2, 1, 1);
    dim3 gridSize(numBlocks,1,1);

    // Create CUDA device variables
    double *dev_A, *dev_G;
    cudaError_t err;
    // Initialize matrix host_A
    for (int i = 0; i < N; i++)
    {
//        host_A[i] = (double) (rand() % 1000000) / 1000.0;
        host_A[i] = (double) i * 1.0;
    }


    // Create device memory and copy from host to device
    err = cudaMalloc((void **) &dev_A, vecSize);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Vector A Malloc: %s\n", cudaGetErrorString(err));

    err =cudaMalloc((void **) &dev_G, vecSize);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Vector G Malloc: %s\n", cudaGetErrorString(err));

    // Copy to device
    gettimeofday(&fullStart, NULL);
    err = cudaMemcpy(dev_A, host_A, vecSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Vector A Copy: %s\n", cudaGetErrorString(err));

    // Perform scan
    gettimeofday(&scanStart, NULL);
    prescan<<<1, N, 2*N*sizeof(double)>>>(dev_G, dev_A, N);
//    prescan<<<grid, threads, 2*N*sizeof(double)>>>(dev_G, dev_A, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Prescan Computation: %s\n", cudaGetErrorString(err));

    // Ensure device is finished
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Device Synchronize: %s\n", cudaGetErrorString(err));

    // Get end of computation time
    gettimeofday(&scanEnd, NULL);

    // Copy result back to host
    cudaMemcpy(host_G, dev_G, vecSize, cudaMemcpyDeviceToHost);
    gettimeofday(&fullEnd, NULL);

    // Free GPU memory
    err = cudaFree(dev_A);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Freeing Vector A: %s\n", cudaGetErrorString(err));

    err = cudaFree(dev_G);
    if (err != cudaSuccess) fprintf(stderr, "ERROR in Freeing Vector G: %s\n", cudaGetErrorString(err));

    // Calculate GPU computation and full runtime
    d_gpuScanTime = myDiffTime(scanStart, scanEnd);
    d_gpuFullTime = myDiffTime(fullStart, fullEnd);

    // CPU Implementation
    gettimeofday(&fullStart, NULL);
    scanCPU(host_C, host_A, N);
    gettimeofday(&fullEnd, NULL);

    // Calculate CPU runtime
    d_cpuTime = myDiffTime(fullStart, fullEnd);



    for (int i = 0; i < N; i++)
    {
        printf("c[%i] = %0.3f, g[%i] = %0.3f\n", i, host_C[i], i, host_G[i]);
//        if (host_C[i] != host_G[i])
//        {
//        	printf("Results do not match! c[%i]=%f, g[%i]=%f\n", i, host_C[i], i, host_G[i]);
//        	break;
//        }
    }

    printf("GPU Time for full run  %i: %f\n", N, d_gpuFullTime);
    printf("GPU Time for scan size %i: %f\n", N, d_gpuScanTime);
    printf("CPU Time for scan size %i: %f\n", N, d_cpuTime);
    free(host_A);
    free(host_G);
    free(host_C);
}

__global__ void prescan(double *g_odata, double *g_idata, int n)
{
    extern  __shared__  double temp[];

    int thid = threadIdx.x;
    if (thid < n)
    {
        int offset = 1;
        temp[2 * thid] = g_idata[2 * thid];
        temp[2 * thid + 1] = g_idata[2 * thid + 1];

        // build sum in place up the tree
        for (int d = n >>= 1; d > 0; d >>= 1)
        {
            __syncthreads();
            if (thid < d)
            {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }

        // clear the last element
        if (thid == 0) temp[n - 1] = 0;

        // traverse down tree & build scan
        for (int d = 1; d < n; d *= 2)
        {
            offset >>= 1;
            __syncthreads();
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            if (thid < d && bi < 2*n)
            {
                double t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();
        g_odata[2 * thid] = temp[2 * thid];
        // write results to device memory
        g_odata[2 * thid + 1] = temp[2 * thid + 1];
    }
}

void scanCPU(double *f_out, double *f_in, int i_n)
{
    f_out[0] = 0;
    for (int i = 1; i < i_n; i++)
        f_out[i] = f_out[i - 1] + f_in[i - 1];

}