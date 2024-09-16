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

#ifndef TIMER_H_
#define TIMER_H_

#include <sys/time.h>
#include <cuda_runtime.h>

struct CPUTimer
{
  int err;
  double start;
  double stop;
  struct timeval Tv;

  CPUTimer()
  {
    start = 0.0;
    stop = 0.0;
    err = 0;

  }

  void Start()
  {
    err = gettimeofday(&Tv, NULL);
    if (!err)
      start = Tv.tv_sec + Tv.tv_usec * 1.0e-6;
  }

  void Stop()
  {
    err = gettimeofday(&Tv, NULL);
    if (!err)
      stop = Tv.tv_sec + Tv.tv_usec * 1.0e-6;
  }

  double Elapsed()
  {
    double elapsed;
    elapsed = stop - start;
    return elapsed; // return elapsed time in seconds
  }
};


struct GPUTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaStream_t streamToRecord;
	bool m_bDisable;

	GPUTimer(cudaStream_t& streamInParam, bool bDisable=true):m_bDisable(bDisable)
	{
		if (m_bDisable)
			return;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		streamToRecord = streamInParam;
	}

	~GPUTimer()
	{
		if (m_bDisable)
			return;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		if (m_bDisable)
			return;

		cudaEventRecord(start, streamToRecord);
	}

	void Stop()
	{
		if (m_bDisable)
			return;

		cudaEventRecord(stop, streamToRecord);
	}

	float Elapsed()
	{
		if (m_bDisable)
			return 0;

		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop); // returns elapsed time in milli second
		return elapsed;
	}
};

#endif


/* Note on usage */

/*
 * CPUTimer cputimer;
 * cputimer.Start();
 * < piece of code to be timed >
 * cputimer.Stop();
 * printf("time taken for piece of code to run = %f sec", cputimer.Elapsed());

 **************************************************************************
 * GPUTimer gputimer;
 * gputimer.Start();
 * kernelToTime<<<...>>>(...);
 * gputimer.Stop();
 * cudaDeviceSynchronize(); // or some blocking call eg. cudaMemcpy(...)
 * printf("time taken for kernel code to run = %f ms", gputimer.Elapsed());

*/

