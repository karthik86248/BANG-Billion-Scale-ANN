#ifndef UTILS_H_
#define UTILS_H_

#include <cuda_runtime.h>
#include <stdio.h>
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if(code != cudaSuccess)
  {
   fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   if (abort) exit(code);
  }
 }

// wrap each API call with the gpuErrchk macro, which will process the return status of the API call it wraps

// gpuAssert can be modified to raise an exception rather than call exit() in a more sophisticated application if it were required.

//sample usage:
  //gpuErrchk(cudaMalloc(&d_a, sizeof(int)*1000000000000000));

/******************************************************************************************************/

// To check for errors in kernel launches, do the following in the cuda code

//kernel<<<1,122222>>>(a);
//gpuErrchk( cudaPeekAtLastError() ); // or gpuErrchk( cudaGetLastError() );
//gpuErrchk( cudaDeviceSynchronize() );
  
#endif


