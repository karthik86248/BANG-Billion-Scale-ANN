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

#include <cstdio>
#include <cstdint>
#include <sys/stat.h>
#include <vector>
#include <map>
#include "bang.h"

#define ROUND_UP(X, Y) \
	((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

using std::string;

#define L 152 // L_search

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
	uint8_t* pIndex;
	off_t size_indexfile;
	unsigned long long ullIndex_Entry_LEN;
	uint8_t* d_compressedVectors;
	float *d_pqTable;
	unsigned  *d_chunksOffset;
	float *d_centroid;
} IndexLoad;

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
								unsigned* d_numQueries,
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
 							unsigned* d_numQueries,
							unsigned long long MEDOID,
							unsigned R);

__global__ void  compute_parent2(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
							unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
							unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
							unsigned* d_mark,
							unsigned* d_iter,
 							unsigned* d_L2ParentIds,
 							unsigned* d_FPSetCoordsList_Counts,
 							unsigned* d_numQueries,
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
						unsigned* d_numQueries,
						unsigned* d_recall);

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

void parANN(int argc, char** argv);

// BFS related helper code
typedef struct _Node
{
  unsigned uNodeID;
  bool bVisited;

} Node; // size = 8 bytes

typedef std::vector<unsigned> NeighbourList;
typedef std::map<unsigned,Node*> NodeIDMap;
void SetupBFS(NodeIDMap& p_mapNodeIDToNode);
void ExitBFS(NodeIDMap& p_mapNodeIDToNode);
void bfs(unsigned uMedoid,
		const unsigned nNodesToDiscover,
		unsigned& visit_counter,
		NodeIDMap& mapNodeIDToNode,
		uint8_t* pGraph,
		unsigned long long ullIndex_Entry_LEN,
		unsigned long long D);


#define __FUNCSIG__ __PRETTY_FUNCTION__

class ANNException {
	public:
		ANNException(const std::string& message, int errorCode);
		ANNException(const std::string& message, int errorCode,
				const std::string& funcSig,
				const std::string& fileName,
				unsigned int       lineNum);

		std::string message() const;

	private:
		int          _errorCode;
		std::string  _message;
		std::string  _funcSig;
		std::string  _fileName;
		unsigned int _lineNum;
};

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
inline void load_bin_impl(std::basic_istream<char>& reader,
		size_t actual_file_size, T*& data, size_t& npts,
		size_t& dim) {
	int npts_i32, dim_i32;
	reader.read((char*) &npts_i32, sizeof(int));
	reader.read((char*) &dim_i32, sizeof(int));
	npts = (unsigned) npts_i32;
	dim = (unsigned) dim_i32;

	std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
		<< std::endl;

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
		exit(1);
	}

	data = new T[npts * dim];
	reader.read((char*) data, npts * dim * sizeof(T));

	std::cout << "Last bytes: "
		<< getValues<T>(data + (npts - 2) * dim, dim);
	std::cout << "Finished reading bin file." << std::endl;


#if 0
	// added by Somesh
	// to get  the centroids' coordinates in the PQ table.

	for (int64_t zz = 0; zz < (int64_t) npts; ++zz) {
		for (int64_t yy = 0; yy < (int64_t) dim; ++yy)
			std::cout << data[zz * (int64_t) dim + yy]
				<< " \n"[yy + 1 == (int64_t) dim];
	}
#endif

}



template<typename T>
inline void load_bin(const std::string& bin_file, T*& data, size_t& npts,
		size_t& dim) {
	// OLS
	//_u64            read_blk_size = 64 * 1024 * 1024;
	// cached_ifstream reader(bin_file, read_blk_size);
	// size_t actual_file_size = reader.get_file_size();
	// END OLS
	std::cout << "Reading bin file " << bin_file.c_str() << " ..."
		<< std::endl;
	std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
	uint64_t      fsize = reader.tellg();
	reader.seekg(0);

	load_bin_impl<T>(reader, fsize, data, npts, dim);
}
#endif
