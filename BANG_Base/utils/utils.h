/* Copyright 2024 Indian Institute Of Technology Hyderbad, India. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Karthik V., Saim Khan, Somesh Singh
//
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


