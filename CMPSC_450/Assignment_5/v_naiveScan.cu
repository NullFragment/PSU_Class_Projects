#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048

__global__ void scan(float *g_odata, float *g_idata, int n);
__global__ void prescan(float *g_odata, float *g_idata, int n);
void scanCPU(float *f_out, float *f_in, int i_n);

double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start, d_end;
	d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
	d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
	return (d_end - d_start);
}

int main() 
{
	float a[N], c[N], g[N];
	timeval start, end;

	float *dev_a, *dev_g;
	int size = N * sizeof(float);

	double d_gpuTime, d_cpuTime;

	// initialize matrices a 
	for (int i = 0; i < N; i++)
	{
		a[i] = (float)(rand() % 1000000) / 1000.0;
		//printf("a[%i] = %f\n", i, a[i]);
	}
	// initialize a and b matrices here
	cudaMalloc((void **) &dev_a, size);
	cudaMalloc((void **) &dev_g, size);

	gettimeofday(&start, NULL);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

	//scan<<<1,N,2*N*sizeof(float)>>>(dev_g, dev_a, N);
	prescan<<<1,N,2*N*sizeof(float)>>>(dev_g, dev_a, N);
	cudaDeviceSynchronize();

	cudaMemcpy(g, dev_g, size, cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);
	d_gpuTime = myDiffTime(start, end);

	gettimeofday(&start, NULL);
	scanCPU(c, a, N);
	
	gettimeofday(&end, NULL);
	d_cpuTime = myDiffTime(start, end);
	

	cudaFree(dev_a); cudaFree(dev_g);

	for (int i = 0; i < N; i++)
	{
		printf("c[%i] = %0.3f, g[%i] = %0.3f\n", i, c[i], i, g[i]);
		//if (c[i] != g[i])
		//{
		//	printf("Results do not match! c[%i]=%f, g[%i]=%f\n", i, c[i], i, g[i]);
		//	break;
		//}
	}

	printf("GPU Time for scan size %i: %f\n", N, d_gpuTime);
	printf("CPU Time for scan size %i: %f\n", N, d_cpuTime);
}

__global__ void scan(float *g_odata, float *g_idata, int n)
{
	extern __shared__ float temp[]; // allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	// Load input into shared memory.
	// This is exclusive scan, so shift right by one
	// and set first element to 0
	temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (thid >= offset)
			temp[pout*n+thid] += temp[pin*n+thid - offset];
		else
			temp[pout*n+thid] = temp[pin*n+thid];

		__syncthreads();
	}
	g_odata[thid] = temp[pout*n+thid]; // write output
}

__global__ void prescan(float *g_odata, float *g_idata, int n) 
{ 
	extern  __shared__  float temp[];
	// allocated on invocation 
	int thid = threadIdx.x; 
	int offset = 1; 
    	temp[2*thid]   = g_idata[2*thid]; 
	// load input into shared memory 
    	temp[2*thid+1] = g_idata[2*thid+1]; 
	for (int d = n>>1; d > 0; d >>= 1) 
	// build sum in place up the tree 
    	{ 
        	__syncthreads(); 
		if (thid < d)    
        	{ 
			int ai = offset*(2*thid+1)-1; 
			int bi = offset*(2*thid+2)-1; 
		    	temp[bi] += temp[ai];         
  		} 
        	offset *= 2; 
    	} 

	if (thid == 0) 
	{ 
		temp[n - 1] = 0; 
	} 

	// clear the last element 
	for (int d = 1; d < n; d *= 2) 
	// traverse down tree & build scan 
    	{ 
        	offset >>= 1; 
        	__syncthreads(); 
		if (thid < d) 
        	{ 
			int ai = offset*(2*thid+1)-1; 
			int bi = offset*(2*thid+2)-1; 
			float t   = temp[ai]; 
            		temp[ai]  = temp[bi]; 
            		temp[bi] += t; 
        	} 
    	} 
    	__syncthreads(); 
    	g_odata[2*thid]   = temp[2*thid]; 
	// write results to device memory 
    	g_odata[2*thid+1] = temp[2*thid+1];  
}
 
void scanCPU(float *f_out, float *f_in, int i_n)
{
	f_out[0] = 0;
	for (int i = 1; i < i_n; i++)
		f_out[i] = f_out[i-1] + f_in[i-1];

}
