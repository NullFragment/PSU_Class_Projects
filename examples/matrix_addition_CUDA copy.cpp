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


__global__ void matrixAdd (int *a, int *b, int *c);
void matrixAddCPU(int *a, int *b, int *c);

int main() 
{
	int a[N*N], b[N*N], c[N*N];
	int *dev_a, *dev_b, *dev_c;
	timeval start, end;

	int size = N * N * sizeof(int);

	// initialize a and b with real values (NOT SHOWN)
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	//gettimeofday(&start, NULL);
	fprintf(stderr, "test1\n");

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	fprintf(stderr, "test2\n");

	//dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
	dim3 threads(BLOCK_DIM, BLOCK_DIM);
	//dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));
	dim3 grid((int)ceil(N/threads.x),(int)ceil(N/threads.y));
	fprintf(stderr, "test3\n");
	//////////////////////////GPU/////////////////////////////////////////
	gettimeofday(&start, NULL);
	fprintf(stderr, "test4\n");
	//matrixAdd<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c);
	matrixAdd<<<grid,threads>>>(dev_a,dev_b,dev_c);
	fprintf(stderr, "test5\n");
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);
	fprintf(stderr, "test6\n");
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	//gettimeofday(&end, NULL);
	printf("GPU Time for %i additions: %f\n", N, myDiffTime(start, end));
	//////////////////////////CPU//////////////////////////////////////////
	//gettimeofday(&start, NULL);
	//matrixAddCPU(a, b, c);
	//gettimeofday(&end, NULL);
	//printf("CPU Time for %i additions: %f\n", N, myDiffTime(start, end));

	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}


__global__ void matrixAdd (int *a, int *b, int *c) 
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * N;
	if (col < N && row < N) 
	{
		c[index] = a[index] + b[index];
	}
}

void matrixAddCPU(int *a, int *b, int *c)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			c[i*N + j] = a[i*N + j] + b[i*N + j];
}
