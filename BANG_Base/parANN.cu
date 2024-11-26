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
#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <unordered_set>
#include <assert.h>
#include <boost/dynamic_bitset.hpp>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "parANN.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_R 64 // Max. node degree

#define PQ_PIVOTS_FILE_SUFFIX "_pq_pivots.bin"
#define PQ_COMPRESSEDVECTORS_FILE_SUFFIX "_pq_compressed.bin"
#define PQ_CHUNK_OFFSETS_FILE_SUFFIX "_pq_pivots.bin_chunk_offsets.bin"
#define PQ_CENTROID_FILE_SUFFIX "_pq_pivots.bin_centroid.bin"
#define GRAPH_INDEX_FILE_SUFFIX "_disk.bin"
#define GRAPH_INDEX_METADATA_FILE_SUFFIX "_disk_metadata.bin"

//#define BF 40009ULL   // size of bloom-filter (per query) with 40009 -> 400 MB
// 4 GB worth of BF, // we have 4 GB headroom in GPU ,
// could be varied for varying recall/QPS in plots (apart from L, L is the typically varied for plots)
#define BF_ENTRIES  399887U    	 // per query, max entries in BF, (prime number)
const uint32_t BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(uint32_t); // 4-byte mem aligned size for actual allocation

#define _DBG_BOUNDS
//#define _DBG_RERANKING
//#define _DBG_ITERATIONS
//#define _DBG_BLOOMFILTER // For BF efficiency testing
int nQueryID = 0;
//#define FREE_AFTERUSE  // Free memory on CPU adn GPU that are no longer required. Not to be used, if doing multiple runs
					   	 // This might impact the performance, as free happens parallel with search.

// Indicates MAX iterations performed. IF there is at least one qeuery that requires neighbour seek
// for a parent, then iteration will occur. There is one initial iteratoion where Medoid is added (outside do-while)
#define MAX_PARENTS_PERQUERY  (L+50) // a #define is needed for shared memory allocation and the dynamic 
									// value uMAX_PARENTS_PERQUERY canot be used


//#define MAX_PARENTS_PERQUERY (20)

// length of each entry in INDEX file 128 bytes (FP) + 4 bytes (degree) + 256 bytes (AdjList for R=64)
// There are N such entries
#define SIZEPARENTLIST  (1+1) // one to indicate present/abset and other the actual parent

// Capabilities
#define ENABLE_GPU_STATS 		0x00000001
#define ENABLE_CACHE_WARMUP 	0x00000002

#define PQ_BITS 8
#define CODEBOOK_SIZE 1 << PQ_BITS

using namespace std;
using Clock = std::chrono::high_resolution_clock;

//texture<uint8_t, 1, cudaReadModeElementType> tex_compressedVectors; // for 1D texture memory


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

uint32_t long long log_message (const char* message)
{
	const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	const std::time_t ctimenow_obj = std::chrono::system_clock::to_time_t(now );
	auto duration = now.time_since_epoch();
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	struct tm *local = localtime(&ctimenow_obj);
	//printf("Today is %s %s\n", ctime(&ctimenow_obj),message);
	cout <<  local->tm_hour <<":" << local->tm_min <<":"<< local->tm_sec << " [ " << millis << " ] : " << message << endl;
	return millis;
}



static IndexLoad objIndexLoad;
static SearchParams objSearchParams;
template<typename T>
void bang_load(char* indexfile_path_prefix)
{
	string diskann_Generatedfiles_path = string(indexfile_path_prefix);
	string pqTable_file = diskann_Generatedfiles_path +  PQ_PIVOTS_FILE_SUFFIX;//string(argv[1]); // Pivot files
	string compressedVector_file = diskann_Generatedfiles_path + PQ_COMPRESSEDVECTORS_FILE_SUFFIX;//string(argv[2]);
	string graphAdjListAndFP_file = diskann_Generatedfiles_path + GRAPH_INDEX_FILE_SUFFIX;//string(argv[3]);
	string graphMetadata_file = diskann_Generatedfiles_path + GRAPH_INDEX_METADATA_FILE_SUFFIX;//string(argv[3]);
    string chunkOffsets_file = diskann_Generatedfiles_path + PQ_CHUNK_OFFSETS_FILE_SUFFIX ;//string(argv[5]);
	string centroid_file = diskann_Generatedfiles_path + PQ_CENTROID_FILE_SUFFIX;////string(argv[2]);

	// Check if files exist
	ifstream in1(pqTable_file, std::ios::binary);
	if(!in1.is_open()){
		printf("Error.. Could not open the PQ Pivots File: %s", pqTable_file.c_str() );
		return;
	}
	cout << pqTable_file << endl;
	ifstream in2(compressedVector_file, std::ios::binary);
	if(!in2.is_open()){
		printf("Error.. Could not open the PQ Compressed Vectors File: %s", compressedVector_file.c_str() );
		return;
	}
	cout << compressedVector_file<< endl;
	
	ifstream in3(graphAdjListAndFP_file, std::ios::binary);
	if(!in3.is_open()){
		printf("Error.. Could not open the Graph Index File: %s", graphAdjListAndFP_file.c_str());
		return;
	}
	cout << graphAdjListAndFP_file<< endl;

	// Reading the Graph Metadata File
	GraphMedataData objGrapMetaData;
	ifstream in5(graphMetadata_file, std::ios::binary);
	if(!in5.is_open()){
		printf("Error.. Could not open the Metadata File: %s\n", graphMetadata_file.c_str());
		return;
	}
	
	// Load Graph Metadata first
	in5.read((char*)&objGrapMetaData, sizeof(GraphMedataData));
	cout << "Metadata : " << objGrapMetaData.ullMedoid << ", " << objGrapMetaData.ulluIndexEntryLen  << ", " << objGrapMetaData.uDatatype  << 
	", " << objGrapMetaData.uDim << ", " << objGrapMetaData.uDegree << ", " << objGrapMetaData.uDatasetSize << endl; 
	
	uint32_t long long INDEX_ENTRY_LEN = objGrapMetaData.ulluIndexEntryLen;
	uint32_t long long MEDOID = objGrapMetaData.ullMedoid;
	//uint32_t uDataType  = objGrapMetaData.uDatatype;
	uint32_t D  = objGrapMetaData.uDim;
	uint32_t R  = objGrapMetaData.uDegree;
	uint32_t N  = objGrapMetaData.uDatasetSize;
	
	objIndexLoad.INDEX_ENTRY_LEN = INDEX_ENTRY_LEN;
	objIndexLoad.MEDOID = MEDOID;
	//objIndexLoad.uDataType = uDataType;
	objIndexLoad.D = D;
	objIndexLoad.R = R;
	objIndexLoad.N = N;


	// Loading PQTable (binary)
	float *pqTable = NULL;
	pqTable = (float*) malloc(sizeof(float) * (CODEBOOK_SIZE * D)); // Contains pivot coordinates
	if (NULL == pqTable)
	{
		printf("Error.. Malloc failed PQ Table\n");
		return;
	}

	in1.seekg(8);
	in1.read((char*)pqTable,sizeof(float)*CODEBOOK_SIZE*D);
	in1.close();

	// Loading Compressed Vector (binary)
//	uint32_t int N = 0;
	in2.read((char*)&N, sizeof(int));
	cout << "No of points = " << N	 << endl;

	uint32_t int uChunks = 0;
	in2.read((char*)&uChunks, sizeof(int));
	cout << "Chunks = " << uChunks << endl;
	objIndexLoad.uChunks = uChunks;

	uint8_t * compressedVectors = NULL;
	compressedVectors = (uint8_t*) malloc(sizeof(uint8_t) * uChunks * N);

	if (NULL == compressedVectors)
	{
		printf("Error.. Malloc failed for Compressed Vectors\n");
		return;
	}

	in2.read((char*)compressedVectors, sizeof(uint8_t)*N*uChunks);
	in2.close();
	// To reduce Peak Host memory usage, loading compressed vectors to CPU and transferring to GPU and releasing host memory
	auto d_compressedVectors = raft::make_device_matrix<uint8_t, uint32_t>(handle, N, uChunks); 	//100M*100 ~10GB
	raft::copy(d_compressedVectors.data_handle(), compressedVectors, (uint32_t long long)(uChunks)*N, handle.get_stream());
	objIndexLoad.d_compressedVectors = d_compressedVectors.data_handle();
	free(compressedVectors);
	compressedVectors = NULL;
	RAFT_LOG_INFO("Transferring Compressed vectors done\n");

	uint8_t* pIndex = NULL;
	off_t size_indexfile = caclulate_filesize(graphAdjListAndFP_file.c_str());

	pIndex = (uint8_t*)malloc(size_indexfile);
	if (NULL == pIndex)
	{
		printf("Error.. Malloc failed for Graph\n");
		return;
	}
	objIndexLoad.pIndex = pIndex;
	int nRetIndex = mlock(pIndex, sizeof(size_indexfile));
	cout << "mlock ret for Index: " << nRetIndex << endl;
	if (nRetIndex)
		perror("Index File");
	log_message("LOAD STARTED");		
	in3.read((char*)pIndex, size_indexfile);
	log_message("LOAD ENDED");		
	in3.close();
	
// Loading chunk offsets
	uint32_t n_chunks = uChunks;
	uint32_t *chunksOffset = (uint32_t*) malloc(sizeof(uint32_t) * (n_chunks+1));
	uint64_t numr = n_chunks + 1;
	uint64_t numc = 1;

	load_bin<uint32_t>(chunkOffsets_file, chunksOffset, numr, numc);	//Import the chunkoffset file


	// Loading centroid coordinates
	float* centroid = nullptr;
	load_bin<float>(centroid_file, centroid, numr, numc);				//Import centroid from centroid file

	// GroundTruth loading done later
	// Sanity test to see if the Index file was loaded properly
	uint32_t* puNeighbour = NULL; // Very first neighbour
	uint32_t* puNeighbour1 = NULL; // Very Last neighbour
	uint32_t *puNumNeighbours = NULL;
	uint32_t *puNumNeighbours1 = NULL;
	// First neighbour calculation
	const uint32_t long long ullIndex_Entry_LEN = INDEX_ENTRY_LEN;
	objIndexLoad.ullIndex_Entry_LEN = ullIndex_Entry_LEN;
	puNumNeighbours = (uint32_t*)(pIndex+((INDEX_ENTRY_LEN*0)+ (sizeof(T)*D) ));
	puNeighbour = puNumNeighbours + 1;
	// Last neighbour calculation
	puNumNeighbours1 = (uint32_t*)(pIndex + ( (ullIndex_Entry_LEN * (N-1)) + (sizeof(T)*D) )) ;
	puNeighbour1 = puNumNeighbours1 + (*puNumNeighbours1) ;
	// Print the first and last neighbour in AdjList
	printf("%u \t %u\n", *puNeighbour, *puNeighbour1);

	assert (*puNeighbour <= N);
	assert (*puNeighbour1 <= N);
	uint32_t medoidID = MEDOID;
	objIndexLoad.medoidID = medoidID;
	
	//printf("pIndex = %p \t indexentrylen = %llu\n", objIndexLoad.pIndex, objIndexLoad.ullIndex_Entry_LEN);
	cout << "medoid is : " << medoidID << "\t"  << endl;

	auto d_pqTable = raft::make_device_matrix<float>(handle, CODEBOOK_SIZE, D);
	auto d_chunksOffset = raft::make_device_vector<uint32_t>(handle, n_chunks+1);
	auto d_centroid = raft::make_device_vector<float>(handle, D);			//4*128 ~512B
	raft::copy(d_centroid.data_handle(), centroid, D, handle.get_stream());
	objIndexLoad.d_centroid = d_centroid.data_handle();
	// transpose pqTable
	float *pqTable_T = NULL; // always float irrespective of T

	pqTable_T = (float*) malloc(sizeof(float) * (CODEBOOK_SIZE * D));
	for(uint32_t row = 0; row < 256; ++row) {
		for(uint32_t col = 0; col < D; ++col) {
			pqTable_T[col* 256 + row] = pqTable[row*D+col];
		}
	}

	// host to device transfer
	raft::copy(d_pqTable.data_handle(), pqTable_T, 256*D, handle.get_stream());
	//gpuErrchk(cudaMemcpy(d_compressedVectors, compressedVectors, (uint32_t long long)(sizeof(uint8_t) * (uint32_t long long)(uChunks)*N),cudaMemcpyHostToDevice));
	raft::copy(d_chunksOffset.data_handle(), chunksOffset, n_chunks+1, handle.get_stream());
	objIndexLoad.d_pqTable = d_pqTable.data_handle();
	objIndexLoad.d_chunksOffset = d_chunksOffset.data_handle();
}
template void bang_load<float>(char* );
template void bang_load<uint8_t>(char* );

void bang_load_c(char* pszPath)
{
	bang_load<uint8_t>(pszPath);
}

void bang_set_searchparams(int recall, int worklist_length, DistFunc nDistFunc)
{
	objSearchParams.recall = recall;
	objSearchParams.worklist_length = worklist_length;
	objSearchParams.uDistFunc = nDistFunc;
}

void bang_set_searchparams_c(int recall, int worklist_length, DistFunc nDistFunc)
{
	objSearchParams.recall = recall;
	objSearchParams.worklist_length = worklist_length;
	objSearchParams.uDistFunc = nDistFunc;
}

template<typename T>
void bang_query(raft::resources handle, T* query_array, int num_queries, 
					result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist ) 
{
	// the eight i/p files (7 bin + 1 txt)
	
	uint32_t numQueries = num_queries;
	uint32_t numThreads_K4 = 1; //	compute_parent
	uint32_t numThreads_K1 = 256;//	populate_pqDist_par
	uint32_t numThreads_K2 = 512;// compute_neighborDist_par
	uint32_t numThreads_K5 = 256;// neighbor_filtering_new
	uint32_t recall_at = objSearchParams.recall; // 10
	uint32_t numCPUthreads = 64;//64 // ToDo: get his dynamically
	uint32_t bitCapabilities = 1;// 1,2,4,8,..
	uint32_t K4_blockSize = 256;

	// Assign Capabilites
	bool bEnableGPUStats = false;
	bool bCacheWarmUp = false;

	if (bitCapabilities & ENABLE_GPU_STATS)
		bEnableGPUStats = true;

	if (bitCapabilities & ENABLE_CACHE_WARMUP)
		bCacheWarmUp = true;

	printf("BF Memory = %u BF_ENTRIES = %u\n",BF_MEMORY, BF_ENTRIES);

	uint32_t long long INDEX_ENTRY_LEN = objIndexLoad.INDEX_ENTRY_LEN;
	uint32_t long long MEDOID = objIndexLoad.MEDOID;
	uint32_t D  = objIndexLoad.D;
	uint32_t R  = objIndexLoad.R;

	T* queriesFP = query_array;

	// CPU Data Structs
	uint32_t *neighbors = NULL;
	uint32_t *numNeighbors_query = NULL;
	uint32_t *parents = NULL;
	const uint32_t long long FPSetCoords_size_bytes = D;

	const uint32_t long long FPSetCoords_rowsize_bytes = FPSetCoords_size_bytes * numQueries;

	const uint32_t long long FPSetCoords_size = D ;

	const uint32_t long long FPSetCoords_rowsize = FPSetCoords_size * numQueries;


	T* FPSetCoordsList = NULL;

	// Note : R+1 is needed because MEDOID is added as additional neighbour in very first neighbour fetching
	auto neighbors = raft::make_pinned_vector<uint32_t>(numQueries*(R+1));
	auto numNeighbors_query = raft::make_pinned_vector<uint32_t>(numQueries);
	auto parents = raft::make_device_vector<uint32_t>(numQueries * SIZEPARENTLIST);


	// Final set of K NNs for eacy query will be collected here (sent by GPU)
	//nearestNeighbours = (uint32_t*)malloc(sizeof(uint32_t) * recall_at * numQueries);
	//nearestNeighbours_dist = (uint32_t*)malloc(sizeof(float) * recall_at * numQueries);
	// Allocate host pinned memory for async memcpy
	// [dataset Dimensions * numQuereis] * numIterations
	


	// GPU Data Structs
	

	auto pool = handle.get_stream_pool();
	// A specific stream to to H2D of the FP vectors
 	cudaStream_t streamFPTransfers = pool.get_stream(0);
 	cudaStream_t streamKernels = pool.get_stream(1);

 	cudaStream_t streamParent = pool.get_stream(2);
 	cudaStream_t streamChildren = pool.get_stream(3);

	// GPU execution times
	double time_transfer = 0.0f;
	double time_K1 = 0.0f;
	double time_B1 = 0.0f;
	double time_B2 = 0.0f;
	double time_neighbor_filtering = 0.0f;

	// CPU execution times
	double fp_set_time_gpu = 0.0f; // GPU side
	double seek_neighbours_time = 0.0f;
	vector<double> time_B1_vec;
	vector<double> time_B2_vec;
	uint32_t numThreads_K3_merge = 2*L;
	assert(numThreads_K3_merge <= 1024);	// Max thread block size
	uint32_t numThreads_K3 = R+1;
	time_B2_vec.push_back(0.0);
	bool nextIter = false;
	uint32_t iter = 1; // Note 1-based not 0


	vector<vector<uint32_t>> final_bestL1;	// Per query vector to store the visited parent and its distance to query point
	final_bestL1.resize(numQueries);
	uint32_t offset =  (R+1);
	// Experimentation with using Shared Meory for PQDist tables (no latency improvement observed)
	// gpuErrchk(cudaFuncSetAttribute(compute_neighborDist_par, cudaFuncAttributeMaxDynamicSharedMemorySize, uChunks * 256 *sizeof(float)));

	// Allocations on GPU
	//gpuErrchk(cudaMalloc(&d_compressedVectors, sizeof(uint8_t) * N * uChunks)); 	//100M*100 ~10GB
	auto d_processed_bit_vec  = raft::make_device_matrix<bool>(handle, BF_MEMORY, numQueries);
	auto d_nextIter = raft::make_device_scalar<bool>(handle, true);

	auto d_pqDistTables = raft::make_device_matrix<float>(handle, 256, objIndexLoad.uChunks*numQueries);

	auto d_queriesFP = raft::make_device_matrix<T>(handle, numQueries, D);
	
	auto d_neighbors_aux = raft::make_device_matrix<uint32_t>(handle, numQueries, (R+1));
	auto d_numNeighbors_query = raft::make_device_vector<uint32_t>(handle, numQueries);

	auto d_neighborsDist_query = raft::make_device_matrix<float>(handle, numQueries, (R+1));
	auto d_neighborsDist_query_aux = raft::make_device_matrix<float>(numQueries, (R+1));

	
	auto d_parents = raft::make_device_matrix<uint32_t>(handle, numQueries, SIZEPARENTLIST);
	auto d_neighbors = raft::make_device_matrix<uint32_t>(handle, numQueries, (R+1));
	auto d_neighbors_temp = raft::make_device_matrix<uint32_t>(handle, numQueries, (R+1));
	auto d_numNeighbors_query_temp = raft::make_device_vector<uint32_t>(handle, numQueries);


	auto d_BestLSets_count = raft::make_device_vector<uint32_t>(numQueries);
	auto d_mark = raft::make_device_vector<uint32_t>(numQueries);			// ~40KB


	gauto d_FPSetCoordsList_Counts = raft::make_device_vector<uint32_t>(numQueries);
	auto d_nearestNeighbours = raft::make_device_matrix<result_ann_t>(handle, numQueries, recall_at);// Dim: [recall_at * numQueries]

	GPUTimer gputimer (streamKernels,!bEnableGPUStats);	// Initiating the GPUTimer class object

	

	// There are 3 stages for free'ing: 1) After transferring to Device (i.e. before search) 2) After the iterations and 3) Before termination
#ifdef FREE_AFTERUSE
	// ToDo : To reduce CPU peak memory, the compressed vectors cna be transferred to GPU first and free'd. Then load the graph on CPU
	free(pqTable_T);
	pqTable_T = NULL;
	free(chunksOffset);
	chunksOffset = NULL;
	sleep(10); // waiting to get the free(compressedVectors) to settle down
#endif

	// Transfer the Medoid (seed parent) default first parent
	uint32_t* L2ParentIds = (uint32_t*)malloc(sizeof(uint32_t) * numQueries);
	uint32_t* FPSetCoordsList_Counts = (uint32_t*)malloc(sizeof(uint32_t) * numQueries);

//do // this is just to run the entire search multiple runs for consistent stats reporting

	// re-read the L value
	uint32_t uWLLen = objSearchParams.worklist_length;

	assert(uWLLen <= L);	// Max thread block size
	numThreads_K3_merge = 2*uWLLen;
	assert(numThreads_K3_merge <= 1024);	// Max thread block size
	uint32_t uMAX_PARENTS_PERQUERY = (uWLLen + 50) ; //Needs to be set with expereince. set it to (2*L) if in doubt

	assert ( uMAX_PARENTS_PERQUERY <= MAX_PARENTS_PERQUERY);
	auto FPSetCoordsList = raft::make_pinned_vector<T>(uMAX_PARENTS_PERQUERY * numQueries * FPSetCoords_size_bytes);
	auto d_FPSetCoordsList = raft::make_device_vector<T>(handle, FPSetCoordsList.size()); // Dim: [numIterations * numQueries]
	auto d_L2distances = raft::make_device_matrix<float>(handle, uMAX_PARENTS_PERQUERY, numQueries); // Dim: [numIterations * numQueries]
	auto d_L2ParentIds = raft::make_device_matrix<uint32_t>(handle, uMAX_PARENTS_PERQUERY, numQueries); // Dim: [numIterations * numQueries]
	auto d_L2distances_aux = raft::make_device_matrix<float>(handle, uMAX_PARENTS_PERQUERY, numQueries); // Dim: [numIterations * numQueries]
	auto d_L2ParentIds_aux = raft::make_device_matrix<uint32_t>(handle, numQueries, uMAX_PARENTS_PERQUERY); // Dim: [numIterations * numQueries]

	auto d_mergedNodes = raft::make_device_vector<uint32_t>(handle, 2*uWLLen);
	auto d_BestLSets_visited = raft::make_device_matrix<bool>(handle, numQueries, uWLLen);
	auto d_BestLSetsDist = raft::make_device_matrix<float>(handle, numQueries, uWLLen);	
	auto d_mergedDist = raft::make_device_matrix<float>(handle, numQueries, 2*uWLLen);
	auto d_BestLSets = raft::make_device_matrix<uint32_t>(handle, numQueries, uWLLen);

	for (int i = 0 ; i < numQueries; i++)
	{
		L2ParentIds[i] = objIndexLoad.medoidID;
		FPSetCoordsList_Counts[i] = 1;
	}

	raft::matrix::fill(handle, d_pqDistTables.view(), 0);
	raft::matrix::fill(handle, d_processed_bit_vec.view(), 0);
	raft::matrix::fill(handle, d_parents.view(), 0);
	raft::matrix::fill(handle, d_BestLSets_count.view(), 0);
	raft::matrix::fill(handle, d_mark.view(), 1);
	// Note: parent for row 0, should be medoidID, but initializing entirely to medoidID just hurt
	raft::copy(d_L2ParentIds.data_handle(), L2ParentIds, numQueries, handle.get_stream());
	raft::copy(d_FPSetCoordsList_Counts.data_handle(), FPSetCoordsList_Counts, numQueries, handle.get_stream());
#ifdef FREE_AFTERUSE
	free(L2ParentIds);
	L2ParentIds = NULL;
	free(FPSetCoordsList_Counts);
	FPSetCoordsList_Counts = NULL;
#endif

//	 start = std::chrono::high_resolution_clock::now();
	// [1] collecting all medoid's neighbors for better locality
	//printf("pIndex = %p \t indexentrylen = %llu Medoid = %u\n", objIndexLoad.pIndex, objIndexLoad.ullIndex_Entry_LEN,objIndexLoad.medoidID);
	uint32_t* puNumNeighbours = (uint32_t*)( objIndexLoad.pIndex + ((objIndexLoad.ullIndex_Entry_LEN * objIndexLoad.medoidID) 
								+ (D*sizeof(T))) );
	//printf("NUm neighbours = %u\n", *puNumNeighbours);
	uint32_t medoidDegree = *puNumNeighbours;
	uint32_t* puNeighbour = puNumNeighbours + 1;
	vector<uint32_t> medoidNeighbors;

	for(uint32_t long long ii = 0; ii < medoidDegree; ++ii)
	{
		medoidNeighbors.push_back(puNeighbour[ii]);
	}


	// [2] Setting the neighbors of medoid as initial candidate neighbors for the query point
	for(uint32_t ii=0; ii < numQueries; ++ii ) {
		neighbors[ii * offset] = objIndexLoad.medoidID; // conside medoid also as initial neighbour
		uint32_t numNeighbors = 1; // medoid is already inserted
		for (uint32_t i = 0; i < medoidDegree; ++i) {
			neighbors[ii * offset + i + 1] = medoidNeighbors[i];
			numNeighbors++;
		}

		numNeighbors_query[ii] = numNeighbors;
		// Copy the medoid's FP vectorsi Async (row 0) iter = 1 here.
		// ToDo: Try to remove the two stage copy and call cudaMemcpyAsync directly here
		memcpy(FPSetCoordsList + ((iter-1) * FPSetCoords_rowsize) + (ii*FPSetCoords_size),
				objIndexLoad. pIndex + (objIndexLoad.ullIndex_Entry_LEN * objIndexLoad.medoidID) ,
				FPSetCoords_size_bytes);
	}
	// 0th row in FPSetCoordsListget copied to GPU in Async fashio
	raft::copy(d_FPSetCoordsList.data_handle() + ((iter-1) * FPSetCoords_rowsize),
					FPSetCoordsList + ((iter-1)* FPSetCoords_rowsize),
					FPSetCoords_rowsize_bytes, streamFPTransfers);

//	stop = std::chrono::high_resolution_clock::now();
//	seek_neighbours_time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000;

//	gputimer.Start();
	//Transfer neighbor IDs and count to GPU
	raft::copy(d_neighbors_temp.data_handle(), neighbors, numQueries*(R+1), handle.get_stream());
	raft::copy(d_numNeighbors_query_temp.data_handle(), numNeighbors_query, numQueries, handle.get_stream());
//	gputimer.Stop();
//	time_transfer += gputimer.Elapsed();

	// Cache Warm-up
	if (bCacheWarmUp)
	{
		// CPU  warm-up
		NodeIDMap mapNodeIDToNode;
		uint32_t visit_counter = 0;
		SetupBFS(mapNodeIDToNode);
		bfs(objIndexLoad.medoidID,1000000,visit_counter,mapNodeIDToNode, objIndexLoad.pIndex,INDEX_ENTRY_LEN,D);
		ExitBFS(mapNodeIDToNode);

		// GPU warm-up
		uint32_t uNeighbours_size = mapNodeIDToNode.size()/R;
		uint32_t *neighbors_warmup = (uint32_t*)malloc(sizeof(uint32_t) * uNeighbours_size);
		auto d_neighbors_warmup = raft::make_device_vector<uint32_t>(uNeighbours_size);
		raft::copy(d_neighbors_warmup,.data_handle() neighbors_warmup, uNeighbours_size, handle.get_stream());
		compute_neighborDist_par_cachewarmup<<< uNeighbours_size, uNeighbours_size/R >>>(d_neighbors_warmup.data_handle(),
		objIndexLoad.d_compressedVectors, objIndexLoad.uChunks,R);
	}

	// Cache Warm-up

	omp_set_num_threads(numCPUthreads);

	auto milliStart = log_message("SEARCH STARTED");
	auto start = std::chrono::high_resolution_clock::now();

	raft::copy(d_queriesFP.data_handle(), queriesFP, (D*numQueries), handle.get_stream());
	auto stop = std::chrono::high_resolution_clock::now();
	time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;
	gputimer.Start();
	/**
	 * [3] Launching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K1" block size.
	 * One thread block is assigned to a query, i.e., "numThreads_K1" threads perform the computation for a query.
	 */
	populate_pqDist_par<<<numQueries, numThreads_K1, (D*(sizeof(float)+sizeof(T))), streamKernels>>> (
							objIndexLoad.d_pqTable, 
							d_pqDistTables.data_handle(), 
							d_queriesFP.data_handle(), 
							objIndexLoad.d_chunksOffset, 
							objIndexLoad.d_centroid, 
							objIndexLoad.uChunks,
							D,
							objSearchParams.uDistFunc == ENUM_DIST_MIPS ? MIPS_EXTRA_DIM : 0);
	gputimer.Stop();
	time_K1 += gputimer.Elapsed();


	// for texture memory
	// for d_compressedVectors
	gputimer.Start();
	raft::matrix::fill(handle, d_numNeighbors_query.view(), 0);
	/** [4] Launching the kernel with "numQueries" number of thread-blocks and block size of 256
	 * One thread block is assigned to a query, i.e., 256 threads perform the computation for a query. The block size has been tuned for performance.
	 */
	 // ToDo: 256 or R
		#ifdef _DBG_BLOOMFILTER
		if (nQueryID == 0)
		{
			printf("Num Neighbours before filtering : %d \n", numNeighbors_query[nQueryID]);
			for (int i = 0; i< numNeighbors_query[nQueryID]; i++)
				printf ("%u, ", neighbors[i] );

			printf("\n");
		}
		#endif

	neighbor_filtering_new<<<numQueries, numThreads_K5, 0, streamKernels >>> (d_neighbors.data_handle(), d_neighbors_temp.data_handle(), d_numNeighbors_query.data_handle() d_numNeighbors_query_temp.data_handle(), d_processed_bit_vec.data_handle(),R);
	gputimer.Stop();
	time_neighbor_filtering += gputimer.Elapsed() ;

	#ifdef _DBG_BLOOMFILTER
	std::set<uint32_t> myVisitedSet; // glgobal set to track all visited nodes
	float myUnvisisted_bf = 0.0 ; // what we found as unvisited
	float myUnvisisted_actual = 0.0; // actual truth of unvisited
	uint32_t uTempNum = 0;
	uint32_t uTempNeighbours[R+1];

	if (nQueryID == 0)
	{
		raft::copy(&uTempNum, d_numNeighbors_query.data_handle() 1, handle.get_stream());
		raft::copy(uTempNeighbours, d_neighbors.data_handle(), uTempNum, handle.get_stream());
		handle.sync_stream();
		printf("Num Neighbours after filtering : %d \n", uTempNum);

		for (int i = 0; i< uTempNum; i++)
		{
			printf ("%u, ", uTempNeighbours[i] );
			myVisitedSet.insert(uTempNeighbours[i]);
		}

		printf("\n");
	}
	#endif

	gputimer.Start();

	//gpuErrchk(raft::matrix::fill(d_neighborsDist_query,0,sizeof(float) * (numQueries*(R+1) )));
	/** [5] Launching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K2" block size.
	 * One thread block is assigned to a query, i.e., "numThreads_K2" threads perform the computation for a query.
	 */
	/*compute_neighborDist_par <<<numQueries, numThreads_K2,uChunks * 256  *sizeof(float), streamKernels >>> (d_neighbors.data_handle(), d_numNeighbors_query.data_handle() d_compressedVectors,
	d_pqDistTables, d_neighborsDist_query);*/

	compute_neighborDist_par <<<numQueries, numThreads_K2,0, streamKernels >>> (d_neighbors.data_handle(), d_numNeighbors_query.data_handle() objIndexLoad.d_compressedVectors,
	d_pqDistTables.data_handle(), d_neighborsDist_query.data_handle(), objIndexLoad.uChunks, R);

	/** [6] Launching the kernel with "numQueries" number of thread-blocks and block size of one.
	 * A songle threads perform the computation for a query.
	 * Note: The  additional arithmetic in the number of thread blocks it to arrive a ceil value. After assigning a required value
	 * to numThreads_K4, the division could leat to a truncated quotient, resulting in lesser threads getting spawned.
	 */
	// technically this is start iteration 1, before this it was iteration 0. finding the parent, is start of iteration
	// Neighbor seek if a given parent, processing neighbors, finding next parent is one iteration.
	// iteration 0 occurs outsied the do while loop
	compute_parent1<<<(numQueries + numThreads_K4 -1 )/numThreads_K4,numThreads_K4,0, streamKernels >>>(d_neighbors.data_handle(), d_numNeighbors_query.data_handle(), d_neighborsDist_query.data_handle(),
			d_BestLSets.data_handle(), d_BestLSetsDist.data_handle(), d_BestLSets_visited.data_handle(), d_parents.data_handle(), d_nextIter, d_BestLSets_count.data_handle(), d_mark.data_handle(),
			iter,d_L2ParentIds.data_handle(),d_FPSetCoordsList_Counts.data_handle(), numQueries,MEDOID, R);
	// [7] Compute distance of MEDOID to Query Points
	gputimer.Stop();
	time_B1_vec.push_back(gputimer.Elapsed());
#ifdef _DBG1
	ofstream fileParents;
	fileParents.open("./parents.bin", ios::binary|ios::out);
#endif





	// Loop until all the query have no new parent
	do
	{
		// Let's wait for all kernels got a chance to execute before we initiate  transfer of the 'd_parents'
		stream_wait(streamKernels);

		start = std::chrono::high_resolution_clock::now();

		// Transfer parent IDs from GPU to CPU
		raft::copy(parents, d_parents.data_handle(), (SIZEPARENTLIST)*numQueries, streamParent);

		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;;
		gputimer.Start();
		/** [8] Launching the kernel with "numQueries" number of thread-blocks and (R+1) block size.
		 * One thread block is assigned to a query, i.e., (R+1) threads perform the computation for a query.
		 * The kernel  sorts an array of size (R+1) per query, so we do not require (R+1) threads per query.
		 */
		compute_BestLSets_par_sort_msort<<<numQueries, numThreads_K3,0, streamKernels >>>(d_neighbors.data_handle(),
															d_neighbors_aux.data_handle(),
															d_numNeighbors_query.data_handle()
															d_neighborsDist_query.data_handle(),
															d_neighborsDist_query_aux.data_handle(),
															d_nextIter.data_handle(), R);

		/** [9] Launching the kernel with "numQueries" number of thread-blocks and (2*L) block size.
		 * One thread block is assigned to a query, i.e., (2*L) threads perform the computation for a query.
		 * The kernel merges, for every query, two arrays each of whose sizes are upperbounded by L, so we do not require more than 2*L threads per query.
		 */
		compute_BestLSets_par_merge<<<numQueries, numThreads_K3_merge, R*sizeof(float), streamKernels >>>(d_neighbors.data_handle(),
										d_numNeighbors_query.data_handle()
										d_neighborsDist_query.data_handle(),
										d_BestLSets.data_handle(),
										d_BestLSetsDist.data_handle(),
										d_BestLSets_visited.data_handle(),
										d_parents.data_handle(),
										iter,
										d_nextIter.data_handle(),
										d_BestLSets_count.data_handle(),
										d_mark.data_handle(),
										uWLLen,
										MEDOID,
										R);
		gputimer.Stop();

		start = std::chrono::high_resolution_clock::now();
		// THis memset is very much required. Though it gets updated but not for queries which don't yield parents
		memset(numNeighbors_query, 0, sizeof(uint32_t) * numQueries	);

		cudaStreamSynchronize(streamParent);

		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;


		start = std::chrono::high_resolution_clock::now();
		#pragma omp parallel
		{
			// NOTE: USE STACK VARIABLES TO AVOID CONCURRENCY ISSUES
			int CPUthreadno =  omp_get_thread_num();
			//#pragma omp for
			/*Collecting the node ids of neighborhood of the parent nodes to send to GPU */
			// Note: Only one parent is supported per query. Becasue offset_neighbors is function of
			// queryID only. To support multiple parents in a query, it should have been dependent on
			// parent number(jj) within a query
			for(uint32_t ii=CPUthreadno; ii < numQueries; ii = ii + numCPUthreads)
			{
				uint32_t numParents = parents[ii*(SIZEPARENTLIST)];
				#ifdef _DBG_BOUNDS

				if (numParents > 1)
					printf("ERROR : numParents of %u iter=%u\n", numParents, iter);
				#endif

				for(uint32_t jj = 1; jj <= numParents; ++jj)
				{
					uint32_t curreParent = parents[ii*(SIZEPARENTLIST)+jj];;
					#ifdef _DBG_VERBOSE
					if (ii == nQueryID)
					{
						printf("\n ==> [%d] CPU side: Parent = %d\n", iter, curreParent);
						//fileParents.write((char*)&(curreParent), sizeof(uint32_t) );
					}
					#endif
					uint32_t offset_neighbors = ii * offset;

					// Copy the Parent's'FP vectors of current query
					// ToDo: Try to remove the two stage copy and call cudaMemcpyAsync directly here
					memcpy(FPSetCoordsList + (iter * FPSetCoords_rowsize) + (ii*FPSetCoords_size),
							objIndexLoad.pIndex + (objIndexLoad.ullIndex_Entry_LEN * curreParent),
							FPSetCoords_size_bytes);

					// Extract Neighbour
					uint32_t *puNumNeighbours = (uint32_t*)(objIndexLoad.pIndex + ((objIndexLoad.ullIndex_Entry_LEN*curreParent)+(sizeof(T)*D)) );;

					#ifdef _DBG_BOUNDS
					// validation
					if (*puNumNeighbours > 64)
						printf("*** ERROR: NumNeighbours %d which is > 64\n",*puNumNeighbours);
					#endif

					memcpy(neighbors+offset_neighbors, puNumNeighbours+1, (*puNumNeighbours) * sizeof(uint32_t));
					numNeighbors_query[ii] = (*puNumNeighbours);
				}
			}
		}
		// Lets wait for the two kernels to complete in parallel.

	    auto stop = std::chrono::high_resolution_clock::now();

		seek_neighbours_time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

		// Lets wait for the two kernels to complete in parallel. Elapsed() is a blocking call
		// Waiting is only for stats and not really need for the functionality
		time_B2_vec.push_back(gputimer.Elapsed());

		start = std::chrono::high_resolution_clock::now();
		// Transfer unfiltered neighbors from CPU to GPU

		raft::copy(d_neighbors_temp.data_handle(), neighbors, numQueries*(R+1),
									streamChildren);

		raft::copy(d_numNeighbors_query_temp.data_handle(), numNeighbors_query, numQueries,
									streamChildren);

		// Transfer the FP vectors also from CPU to GPU in Async fashion
		cudaMemcpyAsync(d_FPSetCoordsList.data_handle() + (iter * FPSetCoords_rowsize),
					FPSetCoordsList + (iter * FPSetCoords_rowsize),
					FPSetCoords_rowsize_bytes, cudaMemcpyHostToDevice, streamFPTransfers);

		cudaMemsetAsync(d_numNeighbors_query.data_handle(), 0, sizeof(uint32_t)*numQueries, streamChildren);
		cudaStreamSynchronize(streamChildren);
		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

		#ifdef _DBG_BLOOMFILTER
		std::set<uint32_t> myNeigbours; //local set to track Neigbours (overwritten on every iteration)

		if (nQueryID == 0)
		{
			printf("Num Neighbours before filtering : %d \n", numNeighbors_query[nQueryID]);
			for (int i = 0; i< numNeighbors_query[nQueryID]; i++)
			{
				printf ("%u, ", neighbors[i] );
				myNeigbours.insert(neighbors[i]);
			}
			printf("\n");
		}
		#endif
		gputimer.Start();

		/** [11] Launching the kernel with "numQueries" number of thread-blocks and block size of 256
		 * One thread block is assigned to a query, i.e., 256 threads perform the computation for a query. The block size has been tuned for performance.
		 */
		neighbor_filtering_new<<<numQueries, numThreads_K5,0, streamKernels >>> (d_neighbors.data_handle(),
																		d_neighbors_temp.data_handle(),
																		d_numNeighbors_query.data_handle()
																		d_numNeighbors_query_temp.data_handle(),
																		d_processed_bit_vec.data_handle(), R);
		gputimer.Stop();
		time_neighbor_filtering += gputimer.Elapsed() ;

		#ifdef _DBG_BLOOMFILTER
		uint32_t uTempNum = 0;
		uint32_t uTempNeighbours[R+1];

		if (nQueryID == 0)
		{
			// Lets filter using std::set and compare with our Bloom Filter
			std::set<int> myResult ;
			set_difference(myNeigbours.begin(),  myNeigbours.end(),
							myVisitedSet.begin(), myVisitedSet.end(),
							std::inserter(myResult, myResult.end()) );

			myVisitedSet.insert(myResult.begin(), myResult.end());

			gpuErrchk(cudaMemcpy(&uTempNum, d_numNeighbors_query.data_handle() sizeof(uint32_t), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(uTempNeighbours, d_neighbors.data_handle(), uTempNum * sizeof(uint32_t), cudaMemcpyDeviceToHost));

			myUnvisisted_bf += uTempNum;
			myUnvisisted_actual += myResult.size();

			float nEfficiency = 1.0;
			if (myResult.size())
				nEfficiency = (float)uTempNum/ (float)myResult.size();

			printf("Num Neighbours after filtering : %d BF efficiency = %f %d\n",  uTempNum, nEfficiency, myResult.size());

			for (int i = 0; i< uTempNum; i++)
				printf ("%u, ", uTempNeighbours[i] );

			printf("\n");
		}
		#endif


		gputimer.Start();
		cudaMemsetAsync(d_neighborsDist_query.data_handle(),0,sizeof(float) * (numQueries*(R+1)), streamKernels );

		/** [12]vLaunching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K2" block size.
		 * One thread block is assigned to a query, i.e., "numThreads_K2" threads perform the computation for a query.
		 */
		/*compute_neighborDist_par <<<numQueries, numThreads_K2,uChunks * 256  *sizeof(float),streamKernels >>> (d_neighbors.data_handle(), d_numNeighbors_query.data_handle() d_compressedVectors,
		d_pqDistTables, d_neighborsDist_query);*/



		compute_neighborDist_par <<<numQueries, numThreads_K2,0,streamKernels >>> (d_neighbors.data_handle(), d_numNeighbors_query.data_handle() objIndexLoad.d_compressedVectors,
		d_pqDistTables.data_handle(), d_neighborsDist_query.data_handle(), objIndexLoad.uChunks, R);

		++iter;

		/** [13] Launching the kernel with "numQueries" number of thread-blocks and block size of one.
	 	* A songle threads perform the computation for a query.
		 */
		compute_parent2<<<(numQueries + numThreads_K4 -1 )/numThreads_K4,numThreads_K4,0, streamKernels >>>(d_neighbors.data_handle(), d_numNeighbors_query.data_handle() d_neighborsDist_query.data_handle(),  d_BestLSets.data_handle(),
				d_BestLSetsDist.data_handle(), d_BestLSets_visited.data_handle(), d_parents.data_handle(), d_nextIter.data_handle(), d_BestLSets_count.data_handle(), d_mark,
				iter, d_L2ParentIds.data_handle(), d_FPSetCoordsList_Counts.data_handle(), numQueries,uWLLen, MEDOID,R);

		gputimer.Stop();
		time_B1_vec.push_back(gputimer.Elapsed());
		start = std::chrono::high_resolution_clock::now();

		raft::copy(&nextIter, d_nextIter.data_handle(), 1, handle.get_stream());  //d_nextIter calculated in compute_parent<<< >>>
		handle.sync_stream();
		// Note: Default Stream operations (cmputation or memory transfers) cannot overlap with operatiosn on other sterrams.
		// Hence, the above call could act as a synchronization mechanism to ensure all kernels are done (next parent ready)
		// before we start seeking neighbours on CPU

	  	//  printf("Iteration = %d\n", iter);

		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

		#ifdef _DBG_BOUNDS
		if (iter == uMAX_PARENTS_PERQUERY-1)
		{
			printf("Error: Iterations crossed the assumed limit. FPSetCoords size overrun \n");
			break;
		}
		#endif
	}
	while(nextIter);

#ifdef _DBG1
	fileParents.close();
#endif



#ifdef FREE_AFTERUSE
	// Free memory that is not used after iterations. This helps to reduce the peak mem usage
	// This can negatively impact search performance
	// GPU
	nRetIndex = munlock(pIndex, size_indexfile);
	if (nRetIndex)
		perror("Index File");

	free(pIndex);
	pIndex = NULL;

	// needed till iterations end
	cudaFreeHost(neighbors);
	neighbors = NULL;
	cudaFreeHost(numNeighbors_query);
	numNeighbors_query = NULL;
	cudaFreeHost(parents);
	parents = NULL;
#endif // #if FREE_AFTERUSE


	//	gputimer.Stop();
	//	time_transfer += gputimer.Elapsed();

	// re-rnking start
#if 1
	gputimer.Start();

	cudaStreamSynchronize(streamFPTransfers);
	// ToDo: Instead of K4_blockSize, MAX_PARENTS_PERQUERY can be used
	compute_L2Dist<<<numQueries, K4_blockSize, D * sizeof(T), handle.get_stream()>>> (d_FPSetCoordsList.data_handle(),
												d_FPSetCoordsList_Counts.data_handle(),
												d_queriesFP.data_handle(),
												d_L2ParentIds.data_handle(),
												d_L2distances.data_handle(),
												numQueries,
												D,
												objSearchParams.uDistFunc == ENUM_DIST_MIPS ? MIPS_EXTRA_DIM : 0);

#ifdef _DBG_RERANKING
{
	cudaDeviceSynchronize();
	float* pTempDists = (float*)malloc(sizeof(float) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempDists, d_L2distances.data_handle(), sizeof(float) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	uint32_t* pTempParents = (uint32_t*)malloc(sizeof(uint32_t) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempParents, d_L2ParentIds.data_handle(), sizeof(uint32_t) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	uint32_t* pTempNumParents = (uint32_t*)malloc(sizeof(uint32_t) * numQueries);
	gpuErrchk(cudaMemcpy(pTempNumParents, d_FPSetCoordsList_Counts.data_handle(), sizeof(uint32_t) * numQueries,
	cudaMemcpyDeviceToHost));

	// print parentIDs and distances for Query 0
	printf ("before Sorting\n");

	for (int i=0; i< pTempNumParents[nQueryID]; i++ )
	{
		printf("Parent = %d \t distance = %f\n", pTempParents[(numQueries*i) + nQueryID ], pTempDists[(numQueries*i) + nQueryID ] );
	}
	free(pTempDists);
	free(pTempParents);
	free(pTempNumParents);
}
#endif
	compute_NearestNeighbours<<<numQueries, MAX_PARENTS_PERQUERY, 0, handle.get_stream()>>> (d_L2ParentIds.data_handle(),
												d_L2ParentIds_aux.data_handle(),
												d_FPSetCoordsList_Counts.data_handle(),
												d_L2distances.data_handle(),
												d_L2distances_aux.data_handle(),
												d_nearestNeighbours.data_handle(),
												numQueries,
												recall_at);



#ifdef _DBG_RERANKING
{
	cudaDeviceSynchronize();
	float* pTempDists = (float*)malloc(sizeof(float) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempDists, d_L2distances.data_handle(), sizeof(uint32_t) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	uint32_t* pTempParents = (uint32_t*)malloc(sizeof(uint32_t) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempParents, d_L2ParentIds.data_handle(), sizeof(uint32_t) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	uint32_t* pTempNumParents = (uint32_t*)malloc(sizeof(uint32_t) * numQueries);
	gpuErrchk(cudaMemcpy(pTempNumParents, d_FPSetCoordsList_Counts.data_handle(), sizeof(uint32_t) * numQueries,
	cudaMemcpyDeviceToHost));

	// print parentIDs and distances for Query

	printf ("After Sorting\n");
	for (int i=0; i< pTempNumParents[nQueryID]; i++ )
	{
		printf("nQueryID = %d Parent = %d \t distance = %f\n", nQueryID, pTempParents[(numQueries*i) + nQueryID ], pTempDists[(numQueries*i) + nQueryID ] );
	}

	free(pTempParents);
	free(pTempDists);
	free(pTempNumParents);
}

#endif
	gputimer.Stop();

	fp_set_time_gpu += gputimer.Elapsed() ;

#ifdef _DBG_RERANKING

		uint32_t* pTempLSet = (uint32_t*)malloc(sizeof(uint32_t) * (L * numQueries));
		gpuErrchk(cudaMemcpy(pTempLSet, d_BestLSets, sizeof(uint32_t) * (L * numQueries),
		cudaMemcpyDeviceToHost));

		float* pTempLSetDist = (float*)malloc(sizeof(float) * (L * numQueries));
		gpuErrchk(cudaMemcpy(pTempLSetDist, d_BestLSetsDist, sizeof(float) * (L * numQueries),
		cudaMemcpyDeviceToHost));

		uint32_t* pTempLSetCount = (uint32_t*)malloc(sizeof(uint32_t) * (  numQueries));
		gpuErrchk(cudaMemcpy(pTempLSetCount, d_BestLSets_count, sizeof(uint32_t) * ( numQueries),
		cudaMemcpyDeviceToHost));
		printf("Best_L_Set : \n");
		for (int i =0; i < pTempLSetCount[nQueryID]; i++)
				printf("Parent = %d \t distance = %f\n", pTempLSet[(nQueryID*L) + i], pTempLSetDist[(nQueryID*L) + i] );
		free(pTempLSet);
		free(pTempLSetDist);
		free(pTempLSetCount);


#endif

	start = std::chrono::high_resolution_clock::now();

	raft::copy(nearestNeighbours, d_nearestNeighbours.data_handle(), recall_at * numQueries,
				handle.get_stream());
	raft::copy(nearestNeighbours_dist, d_L2distances.data_handle(), recall_at * numQueries,
				handle.get_stream());
	handle.sync_stream();				
	stop = std::chrono::high_resolution_clock::now();
	time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

	// re-rnking end
#endif

	auto milliEnd = log_message("SEARCH END");
	cout << "Sizeof result_ann_t: " << sizeof(result_ann_t) << endl;

/*
for (int  nIter = 0; nIter < 2; nIter++)
{
		for (int nIterInner = 0; nIterInner < recall_at; nIterInner++)
		{
			cout << nearestNeighbours[(nIter*recall_at) + nIterInner] << "\t" ;
			//cout << nearestNeighbours[(nIter*recall_at) + nIterInner] << "\t" ;
		}
		cout << endl;
}
*/
#ifdef FREE_AFTERUSE
	free(queriesFP);
	queriesFP = NULL;
	gpuErrchk(cudaFreeHost(FPSetCoordsList));
	//free(nearestNeighbours);
	//nearestNeighbours = NULL;
	//free(nearestNeighbours_dist);
	//nearestNeighbours_dist = NULL;

	cudaStreamDestroy(streamFPTransfers);
	cudaStreamDestroy(streamKernels);
#endif // #if FREE_AFTERUSE

	cout << "iterations = " <<  iter << endl;

	assert(time_B1_vec.size() >= 1);
	float time_B1_avg = time_B1_vec[0];
	time_B1 = time_B1_avg;
	for(uint32_t idx=1; idx<time_B1_vec.size(); ++idx) {
		time_B1_avg = time_B1_avg + (time_B1_vec[idx] - time_B1_avg)/(idx+1); // running average
		time_B1 += time_B1_vec[idx];
	}

	assert(time_B2_vec.size() >= 1);
	float time_B2_avg = time_B2_vec[0];
	time_B2 = time_B2_avg;
	for(uint32_t idx=1; idx<time_B2_vec.size(); ++idx) {
		time_B2_avg = time_B2_avg + (time_B2_vec[idx] - time_B2_avg)/(idx+1); // running average
		time_B2 += time_B2_vec[idx];
	}
#if 1
	cout << "STATS:" << endl;
	cout << "(1) total time_K1 = " << time_K1 << " ms" << endl;
	cout << "(2) avg. time_B1 = " << time_B1_avg << " ms" << endl;
	cout << "(3) total time_B1 = " << time_B1 << " ms" << endl;;
	cout << "(4) avg. time_B2 = " << time_B2_avg << " ms" << endl;
	cout << "(5) total time_B2 = " << time_B2 << " ms" << endl;
	cout << "(6) total neighbor_filtering_time = " << time_neighbor_filtering  << " ms" << endl;
	cout << "(7) total transfer_time (CPU <--> GPU) = " << time_transfer / 1000 << " ms" << endl;
	cout << "(8) total neigbbour seek time = " << seek_neighbours_time /  1000 << " ms" << endl;
	cout << "(9) Time elapsed in L2 Dist computation (GPU)= " << fp_set_time_gpu  << " ms" << endl;
	#ifdef _DBG_BLOOMFILTER
	cout << "10) Bloom Filter Efficiency  = " << (myUnvisisted_bf/myUnvisisted_actual) * 100 << "%" << endl;
	#endif

	double totalTime = time_K1 + time_B1 + time_neighbor_filtering + (time_transfer / 1000) + (seek_neighbours_time / 1000)    ; // in ms
	totalTime += fp_set_time_gpu  ;
	double totalTime_wallclock = milliEnd - milliStart;
	double throughput = (numQueries * 1000.0) / totalTime_wallclock;
	// Note : (5) not included, becasue it is shadowed by (8)
	cout << "Total time = (1) + (3) + (6) + (7) + (8) + (9) = " << totalTime << " ms" << endl;
	cout << "Wall Clock Time = " << totalTime_wallclock << endl;
	cout << "Throughput = " << throughput << " QPS" << endl;
	cout << "Throughput (Exclude Mem Transfers) = " << (numQueries * 1000.0) / (totalTime_wallclock - time_transfer / 1000) << " QPS" << endl;


	
	uint32_t total_size=0;
#ifdef _DBG_BOUNDS
	uint32_t* FPSetCoordsList_Counts_temp = (uint32_t*)malloc(sizeof(uint32_t) * numQueries);

	gpuErrchk(cudaMemcpy(FPSetCoordsList_Counts_temp, d_FPSetCoordsList_Counts.data_handle(), sizeof(uint32_t) * numQueries, cudaMemcpyDeviceToHost ));
	for (int i=0; i< numQueries; i++)
	{
		total_size += FPSetCoordsList_Counts_temp[i];
	}
	free(FPSetCoordsList_Counts_temp);
#endif
	
	cout << "Total candidates " << total_size << endl;
#endif

	// reset counters for next run
	time_K1 = 0.0f;
	time_B1 = 0.0f;
	time_B2 = 0.0f;
	time_B1_vec.clear();
	time_B2_vec.clear();
	time_B1_avg = 0;
	time_B2_avg = 0;
	time_neighbor_filtering = 0;
	time_transfer = 0;
	seek_neighbours_time = 0;
	fp_set_time_gpu = 0;
	iter = 1;



	// Free loop-specific data structs that need to be updated in the next run E.g.L
	gpuErrchk(cudaFreeHost(FPSetCoordsList));

}

template void bang_query<float>(raft::device_resources handle, float* query_file, int num_queries, 
						result_ann_t* nearestNeighbours, float* nearestNeighbours_dist ) ;
template void bang_query<uint8_t>(raft::device_resources handle, uint8_t* query_file, int num_queries, 
						result_ann_t* nearestNeighbours, float* nearestNeighbours_dist ) ;

void bang_query_c(uint8_t* query_array, int num_queries, 
					result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist )
{
	raft::device_resources handle;
	bang_query<uint8_t>(handle, query_array,num_queries, nearestNeighbours, nearestNeighbours_dist );
	/*cout << "From BANG: ";
	for (int  nIter = 0; nIter < 2; nIter++)
	{
		for (int nIterInner = 0; nIterInner < objSearchParams.recall; nIterInner++)
		{
			cout << nearestNeighbours[(nIter*objSearchParams.recall) + nIterInner] << "\t" ;
		}
		cout << endl;
	}*/
}


/**
 * This kernel computes the distances of all centroids from the query vectors, and populates the PQ Distance Tables for all queries
 * @param d_pqTable_T This is the PQ Table of dimension (D x 256), containing the co-ordinates of the centroids.
 *						Note: Size is D * 256 (and not chunks * 256 like PQ distances Matrix)
 * @param d_queriesFP This contains the full-precision coordinates of each of the queries.
 * @param d_pqDistTables This is the concatenated PQ Distance Tables of all the queries.
 * @param d_chunksOffset This stores the offset for indexing into the PQ_Table.
 * @param d_centroid This stores the coordinates of the centroid.
 * @param n_chunks This is the number of chunks in the compressed vector of a node.
 * @param beamWidth This is the beamwidth.
 */
template<typename T>
__global__ void populate_pqDist_par(float* d_pqTable_T, 
									float* d_pqDistTables, 
									T* d_queriesFP, 
									uint32_t* d_chunksOffset, 
									float* d_centroid, 
									uint32_t n_chunks,
									uint32_t long long D,
									uint32_t n_DimAdjust)
 {
 	// [kv] what happens if shared memory overflows in SIFT 1BN
	extern __shared__ char array[];
	T *query_vec = (T*)array;
	float *shm_centroid = (float*)( array + sizeof(T) * D)  ;
	//__shared__ T query_vec[D];
	//__shared__ float shm_centroid[D];
	uint32_t queryID = blockIdx.x;
	uint32_t pqDistTables_offset = queryID * 256 * n_chunks;	// Offset to the beginning of the pqTable entries for this query

	uint32_t gid = queryID * (D - n_DimAdjust);
	uint32_t tid = threadIdx.x;

	//  0 to (D - (n_DimAdjust)) : correspond to original dimensions
	for(uint32_t i= tid; i < D - (n_DimAdjust); i += blockDim.x) {
		query_vec[i] = d_queriesFP[gid + i];
		shm_centroid[i] = d_centroid[i];
	}

	//  D - (n_DimAdjust) to D  : correspond to extra dimension added for MIPS
	if (tid >= D - (n_DimAdjust))
	{
		for(uint32_t i= tid; i < D ; i += blockDim.x) {
			query_vec[i] = 0;
			shm_centroid[i] = d_centroid[i];
		}
	}

	__syncthreads();

	// Calculate and place 256 entries in the pqDistTables array corresponding to the each chunk
	for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
		uint32_t chunk_start = pqDistTables_offset + (256 * chunk);	// Size of pqDistTables = (numQueries * n_chunks * 256)
		for (uint32_t j = d_chunksOffset[chunk]; j < d_chunksOffset[chunk + 1]; j++) { 	// if a chunk contains 4-dimensions
			const float *centers_dim_vec = d_pqTable_T + (256 * j);						// then j=0,1,2,3   4,5,6,7
			// [kv] should 256 be hard-coded or based on the dim
			for (uint32_t idx = tid; idx < 256; idx += blockDim.x) {					//// Memory Coalescing
				float diff = centers_dim_vec[idx] - ((T) query_vec[j] - shm_centroid[j]);

				d_pqDistTables[chunk_start + idx] += (float)(diff * diff);
			}
		}
	}
}

/** This kernel filters out those nodes from the neighborhood of nodes in "d_parents" that  have already been processed before (in a previous iteration) and populates d_neighbors with only the new unseen ones.
 * @param d_neighbors This is populated by the kernel to store the the node ids that will be considered in the next iteration, for each query.
 * @param d_neighbors_temp This stores the node ids of the neighborhood of the nodes fetched from host, for every query.
 * @param d_numNeighbors_query This is populated by the kernel to store the number of nodes in d_neighbors.data_handle(), for every query.
 * @param d_numNeighbors_query_temp This stores the number of neighbors fetched from the host, for every query.
 * @param d_processed_bit_vec This is a boolean vector for keeping track of whether a node is processed on not.
 * @param beamWidth This is the beamwidth.
 */

__global__ void neighbor_filtering_new (uint32_t* d_neighbors,
										uint32_t* d_neighbors_temp,
										uint32_t* d_numNeighbors_query,
										uint32_t* d_numNeighbors_query_temp,
										bool* d_processed_bit_vec,
										uint32_t R) {
	uint32_t queryID = blockIdx.x;
	uint32_t tid = threadIdx.x;

	uint32_t offset_neighbors = queryID * (R+1); //Offset into d_neighbors_temp array
	uint32_t offset_bit_vec = queryID*BF_MEMORY;	//Offset into d_processed_bit_vec vector of bloom filter
	uint32_t numNeighbors = d_numNeighbors_query_temp[queryID];
	bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;

	// For each neighbor in d_neighbors_temp array check if its corresponding bits in the d_processed_bit_vec are already set
	for(uint32_t ii=tid; ii < numNeighbors; ii += blockDim.x ) {
		uint32_t nbr = d_neighbors_temp[offset_neighbors+ii];
		if(!((d_processed_bit_vec_start[hashFn1_d(nbr)]) && (d_processed_bit_vec_start[ hashFn2_d(nbr)])))
		{
			d_processed_bit_vec_start[ hashFn1_d(nbr)] = true;	//Set the bit to true
			d_processed_bit_vec_start[ hashFn2_d(nbr)] = true;	//Set the bit to true
			uint32_t old = atomicAdd(&d_numNeighbors_query[queryID], 1);
			d_neighbors[offset_neighbors + old] = nbr;
		}
	}
}

/*Hash functions used for bloom filters */
__device__ uint32_t hashFn1_d(uint32_t x) {

	// FNV-1a hash
	uint64_t hash = 0xcbf29ce4;
	hash = (hash ^ (x & 0xff)) * 0x01000193 ;
	hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;

	return hash % (BF_ENTRIES );
}

__device__ uint32_t hashFn2_d(uint32_t x) {

	// FNV-1a hash
	uint64_t hash = 0x84222325;
	hash = (hash ^ (x & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 8) & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 16) & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 24) & 0xff)) * 0x1B3;
	return hash % (BF_ENTRIES);
}


/**
 * This kernel computes distance of the neighbour from the medoid using the compressed vectors
 * THe main intent to is to access the compressed vectors of the neighbour and not the distance calculation (which is dummy operation) as such
 ^ One thread is responsible for distance calcuation of one node
 */


__global__ void  compute_neighborDist_par_cachewarmup(uint32_t* d_neighbors,
											uint8_t* d_compressedVectors,
											uint32_t n_chunks,
											uint32_t R) {

	uint32_t tid = threadIdx.x;
	uint32_t queryID = blockIdx.x;
	uint32_t numNeighbors = R;
	uint32_t queryNeighbors_offset  = queryID * (blockDim.x);	// offset into d_neighbors array
	uint32_t* d_neighbors_start = d_neighbors + queryNeighbors_offset;

	for( uint32_t j = tid; j < numNeighbors; j += (blockDim.x) ) { // assign eight threads to a neighbor, within a query

		uint32_t long long compressed_vector_offset = ((uint32_t long long)d_neighbors_start[j])*n_chunks;

		float sum = 0.0f;
		d_compressedVectors += compressed_vector_offset;
		for(uint32_t long long i = tid%8; i < n_chunks; i += 8 ){

			sum += d_compressedVectors[i] ;
		}
	}
}
// The threadblock size is 8*R = 8*64 = 512. Each thread will compute dist on uChunks/8 dimensions
	#define THREADS_PER_NEIGHBOR 8

__global__ void  compute_neighborDist_par(uint32_t* d_neighbors,
											uint32_t* d_numNeighbors_query,
											uint8_t* d_compressedVectors,
											float* d_pqDistTables,
											float*  d_neighborsDist_query,
											uint32_t n_chunks,
											uint32_t R) {
	uint32_t tid = threadIdx.x;
	uint32_t queryID = blockIdx.x;
	uint32_t numNeighbors = d_numNeighbors_query[queryID];
	uint32_t pqDistTables_offset = queryID * 256 * n_chunks;
	float* d_pqDistTables_start = d_pqDistTables + pqDistTables_offset;

	uint32_t queryNeighbors_offset  = queryID * (R+1);	// offset into d_neighbors array
	uint32_t* d_neighbors_start  = d_neighbors + queryNeighbors_offset;
	float* d_neighborsDist_query_start = d_neighborsDist_query + queryNeighbors_offset;
	uint8_t* d_compressedVectors_start = NULL;
	float sum = 0.0f;
	for(uint32_t uIter = tid; uIter < numNeighbors; uIter += blockDim.x)
		d_neighborsDist_query_start[uIter] = 0;

	typedef cub::WarpReduce<float,THREADS_PER_NEIGHBOR> WarpReduce;
	//typedef cub::WarpReduce<float> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage[MAX_R];

	for( uint32_t j = tid/THREADS_PER_NEIGHBOR; j < numNeighbors; j += (blockDim.x)/THREADS_PER_NEIGHBOR ) { // assign eight threads to a neighbor, within a query


		d_compressedVectors_start = d_compressedVectors + ((uint32_t long long)d_neighbors_start[j]) *n_chunks;
		sum = 0.0f;

		for(uint32_t long long i = tid%THREADS_PER_NEIGHBOR; i < n_chunks; i += THREADS_PER_NEIGHBOR ){
			sum += d_pqDistTables_start[(i * 256) + d_compressedVectors_start[i]];
		}

		//atomicAdd(&d_neighborsDist_query_start[j], sum);
		d_neighborsDist_query_start[j] = WarpReduce(temp_storage[j]).Sum(sum);
	}
}
template<typename T>
__global__ void compute_L2Dist (T* d_FPSetCoordsList.data_handle(),
								uint32_t* d_FPSetCoordsList_Counts.data_handle(),
								T* d_queriesFP,
								uint32_t* d_L2ParentIds.data_handle(),
								float* d_L2distances.data_handle(),
								uint32_t numQueries,
								uint32_t long long D,
								uint32_t n_DimAdjust)
{
	extern __shared__ char array[];
	//__shared__ T query_vec[D]; //ToDo can be kept in constant/texture memory
	
	T* query_vec = (T*) array;
	uint32_t queryID = blockIdx.x;
	uint32_t numNodes = d_FPSetCoordsList_Counts[queryID];
	uint32_t tid = threadIdx.x;
	uint32_t gid =  queryID * (D - n_DimAdjust);
	uint32_t numQueries = *d_numQueries;
	// ToDo : see if this can be made global
	const uint32_t long long FPSetCoords_size = D; // * sizeof(T) Note : To be used for array indexing, hence not byte offsets
	const uint32_t long long FPSetCoords_rowsize = FPSetCoords_size * numQueries;

	for(uint32_t ii= tid; ii < D - (n_DimAdjust); ii += blockDim.x) {
		query_vec[ii] = d_queriesFP[gid + ii];
	}

	if (tid >= D - (n_DimAdjust))
	{
		for(uint32_t i= tid; i < D ; i += blockDim.x) {
			query_vec[i] = 0;
			
		}
	}

	__syncthreads();

	// one thread block computes the distances of all the nodes for a query,
	for(uint32_t ii = tid; ii < numNodes; ii += blockDim.x) {
		float L2Dist = 0.0;
		for(uint32_t jj=0; jj < D; ++jj) {
			float diff = d_FPSetCoordsList[(FPSetCoords_rowsize * ii) + (queryID * FPSetCoords_size) + jj] - query_vec[jj];
			L2Dist = L2Dist + (diff * diff);
		}
		d_L2distances[( numQueries * ii) + queryID ] = L2Dist;
	}
}

__global__ void  compute_NearestNeighbours(uint32_t* d_L2ParentIds.data_handle(),
						uint32_t* d_L2ParentIds_aux,
						uint32_t* d_FPSetCoordsList_Counts.data_handle(),
						float* d_L2distances.data_handle(),
						float* d_L2distances_aux,
						result_ann_t* d_nearestNeighbours,
						uint32_t* d_numQueries,
						uint32_t* recall_at)
{
    uint32_t tid = threadIdx.x;
    uint32_t queryID = blockIdx.x;
    uint32_t numNeighbors = d_FPSetCoordsList_Counts[queryID];

    if(tid >= numNeighbors || numNeighbors <= 0) return;

    __shared__ uint32_t shm_pos[MAX_PARENTS_PERQUERY + 1]; // Or MAX_PARENTS_PERQUERY+1?

	// perform parallel merge sort
	for(uint32_t subArraySize=2; subArraySize< 2*numNeighbors; subArraySize *= 2){
		uint32_t subArrayID = tid/subArraySize;
		uint32_t start = subArrayID * subArraySize;
		uint32_t mid = min(start + subArraySize/2, numNeighbors);
		uint32_t end = min(start + subArraySize, numNeighbors);

		if(tid >= start && tid < mid){
			uint32_t lowerBound = lower_bound_d_ex(&d_L2distances[mid * (*d_numQueries)], 0, end-mid, d_L2distances[(tid * (*d_numQueries)) + queryID],
								*d_numQueries, queryID);
			shm_pos[tid] = lowerBound + tid;	// Position for this element
		}

		if(tid >= mid && tid < end)  {
			uint32_t upperBound = upper_bound_d_ex(&d_L2distances[start * (*d_numQueries)], 0, mid-start, d_L2distances[(tid * (*d_numQueries)) + queryID],
								*d_numQueries, queryID);
			shm_pos[tid] = start + (upperBound + tid-mid);	// Position for this element

		}
		__syncthreads();
		__threadfence_block();

		// Copy the neighbors to auxiliary array at their correct position
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			d_L2distances_aux[(shm_pos[i]* (*d_numQueries)) + queryID] = d_L2distances[(i * (*d_numQueries)) + queryID];
			d_L2ParentIds_aux[(shm_pos[i]* (*d_numQueries)) + queryID] = d_L2ParentIds[(i * (*d_numQueries)) + queryID];
		}
		__syncthreads();
		// Copy the auxiliary array to original array
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			d_L2distances[(i * (*d_numQueries)) + queryID] = d_L2distances_aux[(i * (*d_numQueries)) + queryID];
			d_L2ParentIds[(i * (*d_numQueries)) + queryID] = d_L2ParentIds_aux[(i * (*d_numQueries)) + queryID];
		}
		__syncthreads();
	}
	for(uint32_t ii = tid; ii < recall_at; ii += blockDim.x)
	{
		//d_nearestNeighbours[( (*d_numQueries) * ii) + queryID ] = d_L2ParentIds[( (*d_numQueries) * ii) + queryID ];
		d_nearestNeighbours[( (recall_at * queryID) ) + ii ] = d_L2ParentIds[( (*d_numQueries) * ii) + queryID ];
	}
}

 __global__ void  compute_parent1(uint32_t* d_neighbors, uint32_t* d_numNeighbors_query, float* d_neighborsDist_query,
 								uint32_t* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
 								uint32_t* d_parents, bool* d_nextIter, uint32_t* d_BestLSets_count,
 								uint32_t* d_mark,
 								uint32_t iter,
 								uint32_t* d_L2ParentIds,
 								uint32_t* d_FPSetCoordsList_Counts,
 								uint32_t* d_numQueries,
								uint32_t long long MEDOID,
								uint32_t R)
 {
	uint32_t tid = threadIdx.x;
  	uint32_t queryID = (blockIdx.x*blockDim.x) + tid;
	if (queryID >= (*d_numQueries) )
		return;

  	// Single thread is enough to compute the parent (pre-fetched) for a query. So, One ThreadBlock can work
  	// on computing parents for blockDim.x Queries
	uint32_t numNeighbors = d_numNeighbors_query[queryID];
	//uint32_t Best_L_Set_size = d_BestLSets_count[queryID];

	float dist=3.402823E+38;	//assigning max float value
	uint32_t index = 0;
	uint32_t numQueries = *d_numQueries;

	uint32_t offset = queryID *(R+1);

	/*Locate closest neighbor in neighbor list*/
	//  Note: MEDOID is excluded becasue we already fetched its children in the very beginning
	for(uint32_t ii=0; ii < numNeighbors; ++ii) {
		if(d_neighborsDist_query[offset + ii] < dist && d_neighbors[offset+ii]!= MEDOID){
			index=offset+ii;
			dist=d_neighborsDist_query[index];
		}
	}

	uint32_t parentIndex = 0;

//	if(Best_L_Set_size==0){
		parentIndex++;
		d_parents[queryID*(SIZEPARENTLIST)+parentIndex] = d_neighbors[index];
		d_mark[queryID]= d_neighbors[index];
		// Lets mark this nodeID for re-ranking later
		//d_basepoints_parentqueries[d_neighbors[index]] = queryID;
//	}

	// Place the parent in d_parents array if parent is decided and set the next iteration flag to true
	// indicates the numParents used in neighbours seeking step
	d_parents[queryID*(SIZEPARENTLIST)] = parentIndex;
	// Note: parentIndex should ideally be accessed in a synchronized manner.
	// (but only WRITEs (no READs) are there now,   so its ok)
//	if(parentIndex != 0) // parentIndex == 0 is the termination condition for the algorithm.
	{
		*d_nextIter = true;
		// Note: One thread assigned to one Query, so ok to increment (no contention)
		d_FPSetCoordsList_Counts[queryID]++;
		d_L2ParentIds[ (iter * numQueries) + queryID] = d_parents[queryID*(SIZEPARENTLIST)+parentIndex];
	}

}
/** This kernel populates the list of nodes whose neighborhood information has to be fetched in the next iteration, from Best_L_Set or neighbors' list for every query
 * @param d_neighbors This is the concatenated list of node ids for all queries whose distances have to be computed with the respective queries.
 * @param d_numNeighbors_query This stores for every query the number of nodes in d_neighbors.
 * @param d_neighborsDist_query This contains the distances between the node ids in d_neighbors and the respective query for all queries.
 * @param d_BestLSets This contains the concatenated list of the candidate sets for all queries.
 * @param d_BestLSetsDist contains the contatenated list of the distances of the candidate nodes to the respective query, for all queries.
 * @param d_BestLSets_visited This maintains the boolean information about if an entry in the d_BestLSets has been processed or not.
 * @param d_parents This is populated with the list of nodes whose neighborhood information has to be fetched in the next iteration of the outer while loop, for all queries.
 * @param beamWidth This is the beamwidth.
 * @param d_nextIter This maintains boolean flags for all queries, to decide if a query is to processed in the next iteration.
 * @param d_BestLSets_count contains the number of nodes in the d_BestLSets, for every query.
 * @param d_mark Stores the node ID to be marked visited in Best_L_Set in the next invocation of compute_BestLSets_par_merge kernel
 */

 __global__ void  compute_parent2(uint32_t* d_neighbors.data_handle(), uint32_t* d_numNeighbors_query.data_handle() float* d_neighborsDist_query,
 								uint32_t* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
 								uint32_t* d_parents, bool* d_nextIter, uint32_t* d_BestLSets_count,
 								uint32_t* d_mark,
 								uint32_t iter,
 								uint32_t* d_L2ParentIds.data_handle(),
 								uint32_t* d_FPSetCoordsList_Counts.data_handle(),
 								uint32_t* d_numQueries,
								uint32_t uWLLen,
								uint32_t long long MEDOID,
								uint32_t R)
 {
	uint32_t tid = threadIdx.x;
  	uint32_t queryID = (blockIdx.x*blockDim.x) + tid;
	if (queryID >= (*d_numQueries) )
		return;

  	// Single thread is enough to compute the parent (pre-fetched) for a query. So, One ThreadBlock can work
  	// on computing parents for blockDim.x Queries
	uint32_t numNeighbors = d_numNeighbors_query[queryID];
	uint32_t Best_L_Set_size = d_BestLSets_count[queryID];

	float dist=3.402823E+38;	//assigning max float value
	uint32_t index = 0;
	uint32_t numQueries = *d_numQueries;
	uint32_t offset = queryID*(R+1);

	/*Locate closest neighbor in neighbor list*/
	//  Note: MEDOID is excluded becasue we already fetched its children in the very beginning
	for(uint32_t ii=0; ii < numNeighbors; ++ii) {
		if(d_neighborsDist_query[offset + ii] < dist && d_neighbors[offset+ii]!= MEDOID){
			index=offset+ii;
			dist=d_neighborsDist_query[index];
		}
	}
	uint32_t parentOffset = queryID*(SIZEPARENTLIST);
	uint32_t parentIndex = 0;
	uint32_t LOffset = uWLLen*queryID;
	uint32_t LIndex = LOffset;

	/*Compare closest neighbor in neighbor list with first unvisited node in Best_L_Set*/
	for(uint32_t ii=0; ii < Best_L_Set_size; ++ii) {
		LIndex = LOffset + ii;
		if(!d_BestLSets_visited[LIndex]){
			parentIndex++;
			if(dist<d_BestLSetsDist[LIndex]){
				d_parents[parentOffset + parentIndex] = d_neighbors[index];
				d_mark[queryID]= d_neighbors[index];
			}
			else{
				d_BestLSets_visited[LIndex] = true;
				d_parents[parentOffset + parentIndex] = d_BestLSets[LIndex];
			}
			break;
		}
	}

	/*Corner case*/
	if(parentIndex==0 && dist<d_BestLSetsDist[LOffset+Best_L_Set_size-1]){
		parentIndex++;
		d_parents[parentOffset + parentIndex] = d_neighbors[index];
		d_mark[queryID]= d_neighbors[index];
	}


	// Place the parent in d_parents array if parent is decided and set the next iteration flag to true
	// indicates the numParents used in neighbours seeking step
	d_parents[parentOffset] = parentIndex;

	if(parentIndex != 0) // parentIndex == 0 is the termination condition for the algorithm.
	{
		*d_nextIter = true;
		// Note: One thread assigned to one Query, so ok to increment (no contention)
		d_FPSetCoordsList_Counts[queryID]++;
		// At this point the count of parent is one ahead of the iteration number
		d_L2ParentIds[(iter * numQueries) + queryID] = d_parents[parentOffset + parentIndex];
	}
	#ifdef _DBG_ITERATIONS
	else
	{
		if (tid==0)
		{
			// print the
			if (iter == d_FPSetCoordsList_Counts[queryID])
				printf("Query ID = %u and #iteration = %d\n", queryID,d_FPSetCoordsList_Counts[queryID]);
				// No of parents indicates  the number loops/work for that query. Hence printing it as a measure of how long a query runs
		}
	}
	#endif
 }


/** This kernel sorts the neighbors (for every query) in increasing order of their distances to the query, using parallel merge sort
 * The sorted list of neighbors is stored in "d_neighbors" at the end of the kernel.
 * @param d_neighbors This is the concatenated list of node ids for all queries whose distances have to be computed with the respective queries.
 * @param d_neighbors_aux This is an auxiliary array for d_neighbors.data_handle(), used for merge sort.
 * @param d_numNeighbors_query This stores for every query the number of nodes in d_neighbors.
 * @param d_neighborsDist_query This contains the distances between the node ids in d_neighbors and the respective query for all queries.
 * @param d_neighborsDist_query_aux This is an auxiliary array for d_neighborsDist_query, used for merge sort.
 * @param d_nextIter This maintains boolean flags for all queries, to decide if a query is to processed in the next iteration.
 * @param beamWidth This is the beamwidth.
 */
__global__ void  compute_BestLSets_par_sort_msort(uint32_t* d_neighbors.data_handle(),
													uint32_t* d_neighbors_aux,
													uint32_t* d_numNeighbors_query.data_handle()
													float* d_neighborsDist_query,
													float* d_neighborsDist_query_aux,
													bool* d_nextIter,
													uint32_t R) {


	uint32_t tid = threadIdx.x;
    uint32_t queryID = blockIdx.x;
    uint32_t numNeighbors = d_numNeighbors_query[queryID];
    *d_nextIter = false;

    if(tid >= numNeighbors || numNeighbors <= 0) return;

    __shared__ uint32_t shm_pos[MAX_R+1];
    uint32_t offset = queryID*(MAX_R+1);	// Offset into d_neighborsDist_query, d_neighborsDist_query_aux, d_neighbors_aux and d_neighbors arrays

	// perform parallel merge sort
	for(uint32_t subArraySize=2; subArraySize< 2*numNeighbors; subArraySize *= 2){
		uint32_t subArrayID = tid/subArraySize;
		uint32_t start = subArrayID * subArraySize;
		uint32_t mid = min(start + subArraySize/2, numNeighbors);
		uint32_t end = min(start + subArraySize, numNeighbors);

		if(tid >= start && tid < mid){
			uint32_t lowerBound = lower_bound_d(&d_neighborsDist_query[offset + mid], 0, end-mid, d_neighborsDist_query[offset + tid]);
			shm_pos[tid] = lowerBound + tid;	// Position for this element
		}

		if(tid >= mid && tid < end)  {
			uint32_t upperBound = upper_bound_d(&d_neighborsDist_query[offset + start], 0, mid-start, d_neighborsDist_query[offset + tid]);
			shm_pos[tid] = start + (upperBound + tid-mid);	// Position for this element

		}
		__syncthreads();
		__threadfence_block();

		// Copy the neighbors to auxiliary array at their correct position
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			d_neighborsDist_query_aux[offset + shm_pos[i]] = d_neighborsDist_query[offset+i];
			d_neighbors_aux[offset + shm_pos[i]] = d_neighbors[offset+i];
		}
		__syncthreads();
		// Copy the auxiliary array to original array
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			d_neighborsDist_query[offset + i] = d_neighborsDist_query_aux[offset+i];
			d_neighbors[offset + i] = d_neighbors_aux[offset+i];
		}
		__syncthreads();
	}
}




/** This kernel merges the current Best_L_Set (candidate set) with the list of sorted neighbors (in d_neighbors) to update the candidate
  set.
 * @param d_neighbors This is the concatenated list of node ids for all queries whose distances have to be computed with the respective queries.
 * @param d_numNeighbors_query This stores for every query the number of nodes in d_neighbors.
 * @param d_neighborsDist_query This contains the distances between the node ids in d_neighbors and the respective query for all queries.
 * @param d_BestLSets This contains the concatenated list of the candidate sets for all queries.
 * @param d_BestLSetsDist contains the contatenated list of the distances of the candidate nodes to the respective query, for all queries.
 * @param d_BestLSets_visited This maintains the boolean information about if an entry in the d_BestLSets has been processed or not.
 * @param d_parents This is populated with the list of nodes whose neighborhood information has to be fetched in the next iteration of the outer while loop, for all queries.
 * @param beamWidth This is the beamwidth.
 * @param first This is a boolean flag to differentiate between the first call to kernel (outside the while loop) from the rest of the calls.
 * @param d_nextIter This maintains boolean flags for all queries, to decide if a query is to processed in the next iteration.
 * @param d_BestLSets_count contains the number of nodes in the d_BestLSets, for every query.
 */


__global__ void  compute_BestLSets_par_merge(uint32_t* d_neighbors,
											uint32_t* d_numNeighbors_query,
											float* d_neighborsDist_query,
											uint32_t* d_BestLSets,
											float* d_BestLSetsDist,
											bool* d_BestLSets_visited,
											uint32_t* d_parents,
											uint32_t iter,
											bool* d_nextIter,
											uint32_t* d_BestLSets_count,
											uint32_t* d_mark,
											uint32_t uWLLen,
											uint32_t long long MEDOID,
											uint32_t R){
	extern __shared__ char array[];
	float *shm_neighborsDist_query = (float*) array; // R+1 is an upperbound on the number of neighbors
	//__shared__ float shm_neighborsDist_query[R]; // R+1 is an upperbound on the number of neighbors
	__shared__ float shm_currBestLSetsDist[L];
	__shared__ float shm_BestLSetsDist[L];
	__shared__ uint32_t shm_pos[MAX_R+L+1];
	__shared__ uint32_t shm_BestLSets[L];
	__shared__ bool shm_BestLSets_visited[L];

	uint32_t queryID = blockIdx.x;
	uint32_t numNeighbors = d_numNeighbors_query[queryID];
	uint32_t tid = threadIdx.x;

	uint32_t Best_L_Set_size  = 0;
	uint32_t newBest_L_Set_size = d_BestLSets_count[queryID];
	uint32_t nbrsBound;
	uint32_t offset = queryID*(R+1);


	if(numNeighbors > 0){	// If the number of neighbors after filteration is zero then no sense of merging

        if(iter==1){	// If this is the first call to compute_BestLSets_par_merge by this query then initialize d_BestLSets, d_BestLSetsDist...
                nbrsBound = min(numNeighbors,uWLLen);
                for(uint32_t ii=tid; ii < nbrsBound; ii += blockDim.x) {
                        uint32_t nbr =  d_neighbors[offset + ii];
                        d_BestLSets[queryID*uWLLen + tid] = nbr;
                        d_BestLSetsDist[queryID*uWLLen + tid] =   d_neighborsDist_query[offset + ii];
                        d_BestLSets_visited[queryID*uWLLen + tid] = ( nbr == MEDOID);
                }
                __syncthreads();
                newBest_L_Set_size = nbrsBound;
                d_BestLSets_count[queryID] = nbrsBound;
        }
        else {
                Best_L_Set_size = d_BestLSets_count[queryID];
                float maxBestLSetDist = d_BestLSetsDist[uWLLen*queryID+Best_L_Set_size-1];
                for(nbrsBound = 0; nbrsBound < min(uWLLen,numNeighbors); ++nbrsBound) {
                        if(d_neighborsDist_query[offset + nbrsBound] >= maxBestLSetDist){
                                break;
                        }
                }


                nbrsBound = max(nbrsBound, min(uWLLen-Best_L_Set_size, numNeighbors));
                // if both Best_L_Set_size and numNeighbors is less than L, then the max of the two will be the newBest_L_Set_size otherwise it will be L
                newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, uWLLen);

                d_BestLSets_count[queryID] = newBest_L_Set_size;


			/*perform parallel merge */
                for(int i=tid; i < nbrsBound; i += blockDim.x) {
                        shm_neighborsDist_query[i] = d_neighborsDist_query[offset + i];
                }
                for(int i=tid; i < Best_L_Set_size; i += blockDim.x) {
                        shm_currBestLSetsDist[i] = d_BestLSetsDist[uWLLen*queryID+i];
                }
                __syncthreads();
                if(tid < nbrsBound) {
                        shm_pos[tid] =  lower_bound_d(shm_currBestLSetsDist, 0, Best_L_Set_size, shm_neighborsDist_query[tid]) + tid;
                }
                if( tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size)) {
                        shm_pos[tid] =  upper_bound_d(shm_neighborsDist_query, 0, nbrsBound, shm_currBestLSetsDist[tid-nbrsBound]) + (tid-nbrsBound);
                }


                __syncthreads();
                __threadfence_block();



                // all threads of the block have populated the positions array in shared memory
                if(tid < nbrsBound && shm_pos[tid] < newBest_L_Set_size)  {
                        shm_BestLSetsDist[shm_pos[tid]] = shm_neighborsDist_query[tid];
                        shm_BestLSets[shm_pos[tid]] = d_neighbors[offset+tid];
                        shm_BestLSets_visited[shm_pos[tid]] = false;
                }
                if(tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size) && shm_pos[tid] < newBest_L_Set_size) {
                        shm_BestLSetsDist[shm_pos[tid]] = shm_currBestLSetsDist[tid-nbrsBound];
                        shm_BestLSets[shm_pos[tid]] = d_BestLSets[queryID*uWLLen+(tid-nbrsBound)];
                        shm_BestLSets_visited[shm_pos[tid]] = d_BestLSets_visited[queryID*uWLLen+(tid-nbrsBound)];
                }
                __syncthreads();
                __threadfence_block();


                //Copying back from shared memory to device array
                if (tid < newBest_L_Set_size) {
                        d_BestLSetsDist[uWLLen*queryID+tid] = shm_BestLSetsDist[tid];
                        d_BestLSets[uWLLen*queryID+tid] = shm_BestLSets[tid];
                        d_BestLSets_visited[uWLLen*queryID+tid] = shm_BestLSets_visited[tid];
                    }
                __syncthreads();
                __threadfence_block();
        }
	}

		// Mark the node extracted by compute_parent kernel as visited
        for(int i=tid;i<newBest_L_Set_size;i=i+blockDim.x){
                if(d_mark[queryID]==d_BestLSets[uWLLen*queryID+tid])
                        d_BestLSets_visited[uWLLen*queryID+tid]=true;
        }
}



/*Helper device function which returns the position first element in the range [lo,hi), which has a value not less than 'target'.*/
__device__ uint32_t lower_bound_d(float arr[], uint32_t lo, uint32_t hi, float target) {
	uint32_t mid;

	while(lo < hi) {
		mid = (lo + hi)/2;
		float val = arr[mid];
		if (target <= val)
			hi = mid;
		else
			lo = mid + 1;
	}

	return lo;

}

/*Helper device function which returns the position first element in the range (lo,hi], which has a value greater than 'target'.*/
__device__ uint32_t upper_bound_d(float arr[], uint32_t lo, uint32_t hi, float target) {

	uint32_t mid;

	while(lo < hi) {
		mid = (lo + hi)/2;
		float val = arr[mid];
		if (target >= val)
			lo = mid+1;
		else
			hi = mid;
	}

	return lo;
}

/*Helper device function which returns the position first element in the range [lo,hi), which has a value not less than 'target'.*/
__device__ uint32_t lower_bound_d_ex(float arr[], uint32_t lo, uint32_t hi, float target, uint32_t row_size, uint32_t queryID)
{
	uint32_t mid;

	while(lo < hi) {
		mid = (lo + hi)/2;
		float val = arr[(mid*row_size) + queryID];
		if (target <= val)
			hi = mid;
		else
			lo = mid + 1;
	}

	return lo;

}

/*Helper device function which returns the position first element in the range (lo,hi], which has a value greater than 'target'.*/
__device__ uint32_t upper_bound_d_ex(float arr[], uint32_t lo, uint32_t hi, float target,  uint32_t row_size, uint32_t queryID)
{

	uint32_t mid;

	while(lo < hi) {
		mid = (lo + hi)/2;
		float val = arr[(mid*row_size) + queryID];
		if (target >= val)
			lo = mid+1;
		else
			hi = mid;
	}

	return lo;
}

void SetupBFS(NodeIDMap& p_mapNodeIDToNode)
{

}

void ExitBFS(NodeIDMap& p_mapNodeIDToNode)
{
	NodeIDMap::iterator it;

	for(it=p_mapNodeIDToNode.begin(); it!=p_mapNodeIDToNode.end(); ++it)
	{
		free(it->second);
	}
}


// for 100000 Nodes, 1.2 MB memory allocated
template<typename T>
NeighbourList GetNeighbours(uint8_t* pGraph,
							uint32_t curreParent,
							NodeIDMap& mapNodeIDToNode,
							uint32_t long long ullIndex_Entry_LEN,
							uint32_t long long D)
{
	NeighbourList retList;
	// find the children nodes and its degree
	uint32_t long long temp = (ullIndex_Entry_LEN * curreParent) + (D*sizeof(T));
	uint32_t *puNumNeighbours = (uint32_t*)(pGraph + temp );

	for(uint32_t kk = 0; kk < *puNumNeighbours; ++kk)
	{
		Node* pNode = (Node*)malloc(sizeof(Node));
		pNode->uNodeID = *(puNumNeighbours+1+kk);
		pNode->bVisited = false;
		//pNode->nLevel = -1;
		retList.push_back(pNode->uNodeID);
		mapNodeIDToNode[pNode->uNodeID] = pNode;
	}

	return retList;
}

void bfs(uint32_t uMedoid,
		const uint32_t nNodesToDiscover,
		uint32_t& visit_counter,
		NodeIDMap& mapNodeIDToNode,
		uint8_t* pGraph,
		uint32_t long long ullIndex_Entry_LEN,
		uint32_t long long D)
{
	Node* pNode = (Node*)malloc(sizeof(Node));
	pNode->uNodeID = uMedoid;
	pNode->bVisited = true;
	//pNode->nLevel = 0;
	mapNodeIDToNode[pNode->uNodeID] = pNode;
	visit_counter++;
	list<uint32_t> queue;
	queue.push_back(uMedoid);

	while (!queue.empty())
	{
		bool bRet = false;
		uint32_t currentVertex = queue.front();
		queue.pop_front();
		//printf("Visited %d\n", currentVertex);
		NeighbourList listChildres = GetNeighbours<float>(pGraph, currentVertex, mapNodeIDToNode, ullIndex_Entry_LEN, D);

		for (int nIter = 0; nIter < listChildres.size(); nIter++)
		{
			if (mapNodeIDToNode[listChildres[nIter]]->bVisited == true)
				continue;
			mapNodeIDToNode[listChildres[nIter]]->bVisited = true;
			visit_counter++;
			queue.push_back(listChildres[nIter]);

			if (visit_counter == nNodesToDiscover)
			{
				cout << "warm up done : Visited counter:" << visit_counter << endl;
				bRet = true;
				break;
			}
		}
		if (bRet)
			break;
	}
}








