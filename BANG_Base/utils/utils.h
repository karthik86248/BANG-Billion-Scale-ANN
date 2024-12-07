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


#include <unistd.h>
#include <stdio.h>
#include <chrono>
#include <fcntl.h>


#include <cuda_runtime.h>

#include <iostream>

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if(code != cudaSuccess)
  {
   fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   if (abort) exit(code);
  }
 }

off_t caclulate_filesize(const char* chFileName)
{
	int fd = -1;

	if ((fd = open(chFileName, O_RDONLY , (mode_t)0 )) == -1)
	{
		perror("Error opening file for writing");
		exit(EXIT_FAILURE);
	}

	long cur_pos = 0L;
	off_t file_size = 0L;

	// file would be opened with file offset set to the very beginning
	cur_pos = lseek(fd, 0, SEEK_CUR);
	file_size = lseek(fd, 0, SEEK_END);
	lseek(fd, cur_pos, SEEK_CUR);
	close(fd);
	return file_size;
}

unsigned long long log_message (const char* message)
{
	const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	const std::time_t ctimenow_obj = std::chrono::system_clock::to_time_t(now );
	auto duration = now.time_since_epoch();
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	struct tm *local = localtime(&ctimenow_obj);
	std::cout <<  local->tm_hour <<":" << local->tm_min <<":"<< local->tm_sec << " [ " << millis << " ] : " << message << std::endl;
	return millis;
}


#endif


