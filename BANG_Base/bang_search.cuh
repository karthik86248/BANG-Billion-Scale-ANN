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
#ifndef PARANN_H_
#define PARANN_H_

//#include <cstdio>
#include <cuda_runtime.h>
#include "bang.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>


#define ROUND_UP(X, Y) \
	((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)



typedef enum _DataTypeEnum
{
	ENUM_UNSIGNED_INT8 = 0,
	ENUM_UNSIGNED_INT32,
	ENUM_FLOAT

} DataTypeEnum;


typedef struct __attribute__((__packed__)) _GraphMedatadata
{
	unsigned long long ullMedoid;
	unsigned long long ulluIndexEntryLen;
	int uDatatype;
	unsigned uDim; // no of dimensions
	unsigned uDegree; // no of dimensions
	unsigned uDatasetSize;
} GraphMedataData; //32 bytes


typedef struct _IndexLoad
{
	//unsigned long long INDEX_ENTRY_LEN ;
	unsigned long long MEDOID ;
	unsigned uDataType  ;
	unsigned D  ;
	unsigned R  ;
	unsigned N  ;
	unsigned int uChunks;
	//unsigned medoidID;

	off_t size_indexfile;
	unsigned long long ullIndex_Entry_LEN;

	// Device Memory allocations
	uint8_t* d_compressedVectors;
	float *d_pqTable;
	unsigned  *d_chunksOffset;
	float *d_centroid;

	// Host Memory allocation
	uint8_t* pIndex;
} IndexLoad;


typedef struct _GPUInstance
{
	// Kernel threadblock sizes
	unsigned numThreads_K4 ; //	compute_parent kernel
	unsigned numThreads_K1 ;//	populate_pqDist_par
	unsigned numThreads_K2 ;// compute_neighborDist_par
	unsigned numThreads_K3_merge ;
	unsigned numThreads_K3  ;
	unsigned K4_blockSize ;
	unsigned numThreads_K5; // neighbor_filtering_new
	// Device Memory
	void *d_queriesFP ; // Input
	result_ann_t *d_nearestNeighbours = NULL; // The final output of ANNs
	float *d_pqDistTables;
	float *d_BestLSetsDist;
	unsigned *d_BestLSets_count ;
	unsigned *d_BestLSets ;
	bool *d_BestLSets_visited ;
	unsigned *d_parents ;
	float *d_neighborsDist_query ;
	float *d_neighborsDist_query_aux ;
	unsigned *d_neighbors ;
	unsigned *d_neighbors_aux ;
	unsigned *d_numNeighbors_query ;
	unsigned *d_neighbors_temp ;
	unsigned *d_numNeighbors_query_temp ;
	unsigned *d_iter ;
	unsigned *d_mark ;
	bool *d_nextIter ;
	bool *d_processed_bit_vec ;

	// The FP vectors corresponding to each candidate is fetched asynchronously for all the queries in the iteration.
	// This FP vectors are used in the Re-ranking step at the endo of search iterations
	// A 2D array is used to store the FP vectors.

	// 2D array format
	// [FP for Q1] [FP for Q2]...[FP for QN]
	// [FP for Q1] [FP for Q2]...[FP for QN]
	// ..
	// [FP for Q1] [FP for Q2]...[FP for QN] total M such entries (rows)

	// Every iteration: [1 * numQuereis] row added
	// Dimensoins of 2D array : [numIterations * numQueries]
	// numIterations upper bound is MAX_PARENTS_PERQUERY
	void* d_FPSetCoordsList;
	unsigned* d_FPSetCoordsList_Counts;
	float* d_L2distances ; // M x N dimensions
	unsigned* d_L2ParentIds ; // // M x N dimensions
	float* d_L2distances_aux ; // M x N dimensions
	unsigned* d_L2ParentIds_aux ; // // M x N dimensions

	//  Specific Streams for
 	cudaStream_t streamFPTransfers; // H2D
 	cudaStream_t streamParent; // D2H
 	cudaStream_t streamChildren; // H2D
	cudaStream_t streamKernels; // kernels executions

}GPUInstance;


typedef struct _HostInstance
{
	unsigned numCPUthreads; //64 // ToDo: get the core count dynamically from the platform
	unsigned *parents = NULL;
	unsigned *neighbors = NULL;
	unsigned *numNeighbors_query = NULL;
	//unsigned *L2ParentIds;
	//unsigned* FPSetCoordsList_Counts;
	void* FPSetCoordsList;
	unsigned long long FPSetCoords_size_bytes ;
	unsigned long long FPSetCoords_rowsize_bytes;
	unsigned long long FPSetCoords_size ; // no of entries in one vector
	unsigned long long FPSetCoords_rowsize; // no of entries in one row of the FPSetCoordsList matrix
	
}HostInstance;

typedef struct _SearchParams
{
	int recall;
	int worklist_length;
	unsigned uDistFunc;
} SearchParams;

template<typename T>
__global__ void populate_pqDist_par(float *d_pqTable,
									float* d_pqDistTables,
									T* d_queriesFP,
									unsigned* d_chunksOffset,
									float* d_centroid,
									unsigned n_chunks,
									unsigned long long D,
									unsigned n_DimAdjust=0);

template<typename T>
__global__ void compute_L2Dist (T* d_FPSetCoordsList,
								unsigned* d_FPSetCoordsList_Counts,
								T* d_queriesFP,
								unsigned* d_L2ParentIds,
								float* d_L2distances,
								unsigned d_numQueries,
								unsigned long long D,
								unsigned n_DimAdjust=0);


__global__ void  compute_neighborDist_par(unsigned* d_neighbors,
											unsigned* d_numNeighbors_query,
											uint8_t* d_compressedVectors,
											float* d_pqDistTables,
											float*  d_neighborsDist_query,
											unsigned n_chunks,
											unsigned R);

__global__ void  compute_parent1(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
							unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
							unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
							unsigned* d_mark,
							unsigned* d_iter,
 							unsigned* d_L2ParentIds,
 							unsigned* d_FPSetCoordsList_Counts,
 							unsigned d_numQueries,
							unsigned long long MEDOID,
							unsigned R);

__global__ void  compute_parent2(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
							unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
							unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
							unsigned* d_mark,
							unsigned* d_iter,
 							unsigned* d_L2ParentIds,
 							unsigned* d_FPSetCoordsList_Counts,
 							unsigned d_numQueries,
							unsigned uWLLen,
							unsigned long long MEDOID,
							unsigned R);

__global__ void  compute_BestLSets_par_sort_msort(unsigned* d_neighbors,
													unsigned* d_neighbors_aux,
													unsigned* d_neighbors_offset,
													float* d_neighborsDist_query,
													float* d_neighborsDist_query_aux,
													bool* d_nextIter,
													unsigned R);


__global__ void  compute_BestLSets_par_merge(unsigned* d_neighbors,
												unsigned* d_numNeighbors_query,
												float* d_neighborsDist_query,
												 unsigned* d_BestLSets,
												 float* d_BestLSetsDist,
												 bool* d_BestLSets_visited,
												 unsigned* d_parents,
												unsigned iter,
												bool* d_nextIter,
												unsigned* d_BestLSets_count,
												unsigned* d_mark,
												unsigned uWLLen,
												unsigned long long MEDOID,
												unsigned R);

__global__ void neighbor_filtering_new (unsigned* d_neighbors,
									unsigned* d_neighbors_temp,
									unsigned* d_numNeighbors_query,
									unsigned* d_numNeighbors_query_temp,
									bool* d_processed_bit_vec,
									unsigned R);



__global__ void  compute_NearestNeighbours(unsigned* d_L2ParentIds,
						unsigned* d_L2ParentIds_aux,
						unsigned* d_FPSetCoordsList_Counts,
						float* d_L2distances,
						float* d_L2distances_aux,
						result_ann_t* d_nearestNeighbours,
						unsigned d_numQueries,
						unsigned d_recall);

__global__ void  compute_neighborDist_par_cachewarmup(unsigned* d_neighbors,
											uint8_t* d_compressedVectors,
											unsigned n_chunks,
											unsigned R);

__device__ unsigned upper_bound_d(float arr[], unsigned lo, unsigned hi, float target);

__device__ unsigned lower_bound_d(float arr[], unsigned lo, unsigned hi, float target);

__device__ unsigned lower_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target, unsigned row_size, unsigned queryID);

__device__ unsigned upper_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target,  unsigned row_size, unsigned queryID);

__device__ unsigned hashFn1_d(unsigned x);
__device__ unsigned hashFn2_d(unsigned x);





template<typename T>
inline std::string getValues(T* data, size_t num) {
	std::stringstream stream;
	stream << "[";
	for (size_t i = 0; i < num; i++) {
		stream << std::to_string(data[i]) << ",";
	}
	stream << "]" << std::endl;

	return stream.str();
}

template<typename T>
inline bool load_bin_impl(std::basic_istream<char>& reader,
		size_t actual_file_size, T*& data, size_t& npts,
		size_t& dim) {
	int npts_i32, dim_i32;
	reader.read((char*) &npts_i32, sizeof(int));
	reader.read((char*) &dim_i32, sizeof(int));
	npts = (unsigned) npts_i32;
	dim = (unsigned) dim_i32;

	std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

	size_t expected_actual_file_size =
		npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
	if (actual_file_size != expected_actual_file_size) {
		std::stringstream stream;
		stream << "Error. File size mismatch. Actual size is " << actual_file_size
			<< " while expected size is  " << expected_actual_file_size
			<< " npts = " << npts << " dim = " << dim
			<< " size of <T>= " << sizeof(T) << std::endl;
		std::cout << stream.str();
		//throw std::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
		//                           __LINE__);
		return false ;
	}

	data = new T[npts * dim];
	reader.read((char*) data, npts * dim * sizeof(T));

	//std::cout << "Last bytes: " 	<< getValues<T>(data + (npts - 2) * dim, dim);
	std::cout << "Finished reading bin file." << std::endl;
	return true;
}



template<typename T>
inline bool load_bin(const std::string& bin_file, T*& data, size_t& npts,
		size_t& dim) {

	std::cout << "Reading bin file " << bin_file.c_str() << " ..." 	<< std::endl;
	std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
	if (reader.fail())
	{
		std::cout << "Reading bin file " << bin_file.c_str() << " failed. Error:" 	<< strerror(errno) << std::endl;
		return false;
	}
		
	uint64_t      fsize = reader.tellg();
	reader.seekg(0);

	return load_bin_impl<T>(reader, fsize, data, npts, dim);
}
#endif // PARANN_H_
