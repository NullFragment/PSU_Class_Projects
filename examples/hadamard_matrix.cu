#include <stdio.h>
#include <sys/time.h>

__global__ void hadamardProduct (int *a, int *b, int *c, int N);


int main()
{
	int N = 32;
	int BLOCK_DIM = 32;
	int size = N * N * sizeof(int);
	int *a = (int *)calloc(N*N, sizeof(int));
	int *b = (int *)calloc(N*N, sizeof(int));
	int *c = (int *)calloc(N*N, sizeof(int));
	int *dev_a = NULL, *dev_b  = NULL, *dev_c  = NULL;

	for(int i = 0; i < N*N; i++)
	{
	    a[i] = 2;
	    b[i] = 7;
	}

	// initialize a and b with real values (NOT SHOWN)
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	//gettimeofday(&start, NULL);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	dim3 grid((int)ceil(N/threads.x),(int)ceil(N/threads.y));

	hadamardProduct<<<grid,threads>>>(dev_a,dev_b,dev_c, N);
	cudaDeviceSynchronize();

	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
	    for(int j = 0; j < N; j++)
	    {
		printf("%d ", c[i*j]);
	    }
	    printf("\n");
	}
	//gettimeofday(&end, NULL);
	printf("GPU for %i additions\n", N);


	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}


__global__ void hadamardProduct (int *a, int *b, int *c, int N)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * N;
	if (col < N && row < N)
	{
		c[index] = a[index] * b[index];
	}
}
