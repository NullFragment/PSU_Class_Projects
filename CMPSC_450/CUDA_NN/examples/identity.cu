__global__ void initIdentityGPU(int **devMatrix, int numR, int numC) {
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if(y < numR && x < numC) {
          if(x == y)
              devMatrix[y][x] = 1;
          else
              devMatrix[y][x] = 0;
    }
}

dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
dim3 gridDim((numC + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (numR + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
initIdentityGPU<<<gridDim, blockDim>>>(matrix, numR, numC);
