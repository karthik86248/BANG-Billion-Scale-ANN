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

