#include <stdio.h>
#include <sys/time.h>

#define	N 1024
#define BLOCK_DIM 1024

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

	//free the space
	cudaFree(dev_a); 
	cudaFree(dev_b); 
	cudaFree(dev_c);
}
