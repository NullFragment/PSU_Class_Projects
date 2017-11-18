#include <stdio.h>
#include <sys/time.h>

#define	N 1024
#define BLOCK_DIM 1024

double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start, d_end;
	d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
	d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
	return (d_end - d_start);
}


__global__ void
matrixAdd(const float *A, const float *B, float *C, int numElements)
{
	//referred to matrix multiplication code from NVDIA
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
     	C[i] = A[i] + B[i];
   }
}

int main() 
{
	//declaration
	size_t size = N * N * sizeof(float);

	float *a = (float *)malloc(size);
	float *b = (float *)malloc(size);
	float *c = (float *)malloc(size);

	float *dev_a = NULL;
	float *dev_b = NULL;
	float *dev_c = NULL;
	timeval start, end;

	//allocating space
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	//preparing parameter for CUDA
	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	dim3 grid((int)ceil(N/threads.x),(int)ceil(N/threads.y));


	//////////////////////////GPU/////////////////////////////////////////
	gettimeofday(&start, NULL);

	matrixAdd<<<grid,threads>>>(dev_a,dev_b,dev_c,N*N);
	cudaDeviceSynchronize();

	gettimeofday(&end, NULL);

	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	printf("GPU Time for %i additions: %f\n", N, myDiffTime(start, end));

	//free the space
	cudaFree(dev_a); 
	cudaFree(dev_b); 
	cudaFree(dev_c);
}
