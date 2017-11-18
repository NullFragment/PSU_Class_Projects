// Compile using nvcc <file> -lcublas -o <output>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


/**
  This function will initialize memory on GPU for matrices and copy values
  for matrix operation
*/
// void MatrixInitCUDA

/**
  This function will initialize memory on GPU for vectors and copy values
  for vector operations
*/
// void VectorInitCUDA


/**
  This function will call the initialize function then call
  cublasSgemm function, copy result back to host and free GPU memory
*/
//float matrixMultiply


/**
  This function will call vector init function,
  perform vector multiplication, then free GPU memory
*/
//float vectorMultiply

int main()
{
  // Create memory for Layer 1, Layer 2, Layer 3 vectors
  //float *layer1 = malloc(784*sizeof(floats)))
  // Create memory for Weight 1->2, Weight 2->3 matrices


  // Layer 1 will read from file for input (X) values
  // Layer 2 and 3 will be calculated
}
