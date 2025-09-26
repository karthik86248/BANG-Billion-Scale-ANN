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

#define R 64 // Max. node degree

//#define OLD_MERGE 

//#define BF 40009ULL   // size of bloom-filter (per query) with 40009 -> 400 MB
// 4 GB worth of BF, // we have 4 GB headroom in GPU ,
// could be varied for varying recall/QPS in plots (apart from L, L is the typically varied for plots)
#define BF_ENTRIES 399887U   	 // per query, max entries in BF, (prime number)
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned); // 4-byte mem aligned size for actual allocation


//#define _DBG
//#define _DBG2
int nQueryID = 0;
//#define FREE_AFTERUSE  // Free memory on CPU adn GPU that are no longer required.
					   // This might impact the performance, as free happens parallel with search

// Indicates MAX iterations performed. IF there is at least one qeuery that requires neighbour seek
// for a parent, then iteration will occur. There is one initial iteratoion where Medoid is added (outside do-while)
#define MAX_PARENTS_PERQUERY (4*L+20) // Needs to be set with expereince. set it to (2*L) if in doubt

// length of each entry in INDEX file 128 bytes (FP) + 4 bytes (degree) + 256 bytes (AdjList for R=64)
// There are N such entries
#define SIZEPARENTLIST  (1+1) // one to indicate present/abset and other the actual parent

// Capabilities
#define ENABLE_GPU_STATS 	0x00000001
#define ENABLE_CACHE_WARMUP 	0x00000002

using namespace std;
using Clock = std::chrono::high_resolution_clock;

//texture<uint8_t, 1, cudaReadModeElementType> tex_compressedVectors; // for 1D texture memory
const unsigned long long ullIndex_Entry_LEN = INDEX_ENTRY_LEN;

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
	auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
	struct tm *local = localtime(&ctimenow_obj);
	//printf("Today is %s %s\n", ctime(&ctimenow_obj),message);
	cout <<  local->tm_hour <<":" << local->tm_min <<":"<< local->tm_sec << " [ " << nanos << " ] : " << message << endl;
	return nanos;
}

void parANN(int argc, char** argv) {
	// the eight i/p files (7 bin + 1 txt)
	string pqTable_file = string(argv[1]); // Pivot files
	string compressedVector_file = string(argv[2]);
	string graphAdjListAndFP_file = string(argv[3]);
	string queryPointsFP_file = string(argv[4]);
	string chunkOffsets_file = string(argv[5]);
	string centroid_file = string(argv[6]);
	string truthset_bin = string(argv[7]);
	unsigned numQueries = atol(argv[8]);
	unsigned numThreads_K4 = atol(argv[9]); // 1	compute_parent
	unsigned numThreads_K1 = atol(argv[10]); //256	populate_pqDist_par
	unsigned numThreads_K2 = atol(argv[11]); // 512 compute_neighborDist_par
	unsigned numThreads_K5 = atol(argv[12]); // 256	neighbor_filtering_new
	unsigned recall_at = std::atoi(argv[13]);// 10
	unsigned numCPUthreads = atoi(argv[14]); // 64
	unsigned bitCapabilities = atoi(argv[15]); // 1,2,4,8,..

	// Assign Capabilites
	bool bEnableGPUStats = false;
	bool bCacheWarmUp = false;

	if (bitCapabilities & ENABLE_GPU_STATS)
		bEnableGPUStats = true;

	if (bitCapabilities & ENABLE_CACHE_WARMUP)
		bCacheWarmUp = true;

	unsigned medoidID = MEDOID;
	//unsigned K4_blockSize = 256;

	cout << medoidID << "\t" << numQueries << endl;

	printf("BF Memory = %u BF_ENTRIES = %u\n",BF_MEMORY, BF_ENTRIES);

	// Check if files exist
	ifstream in1(pqTable_file, std::ios::binary);
	if(!in1.is_open()){
		printf("Error.. Could not open the file1..");
		return;
	}
	#if NOTNECESSARY
	ifstream in2(compressedVector_file, std::ios::binary);
	if(!in2.is_open()){
		printf("Error.. Could not open the file2..");
		return;
	}
	#endif

	ifstream in3(graphAdjListAndFP_file, std::ios::binary);
	if(!in3.is_open()){
		printf("Error.. Could not open the file..");
		return;
	}

	ifstream in4(queryPointsFP_file, std::ios::binary);
	if(!in4.is_open()){
		printf("Error.. Could not open the file4..");
		return;
	}


	// Loading PQTable (binary)
	float *pqTable = NULL;
	pqTable = (float*) malloc(sizeof(float) * (256 * D)); // Contains pivot coordinates
	if (NULL == pqTable)
	{
		printf("Error.. Malloc failed PQ Table\n");
		return;
	}

	in1.seekg(8);
	in1.read((char*)pqTable,sizeof(float)*256*D);
	in1.close();

#if NOTNECESSARY
	// Loading Compressed Vector (binary)
	uint8_t * compressedVectors = NULL;
	compressedVectors = (uint8_t*) malloc(sizeof(uint8_t) * CHUNKS * N);

	if (NULL == compressedVectors)
	{
		printf("Error.. Malloc failed for Compressed Vectors\n");
		return;
	}


	in2.seekg(8);
	in2.read((char*)compressedVectors, sizeof(uint8_t)*N*CHUNKS);
	in2.close();
#endif
	uint8_t* pIndex = NULL;
	off_t size_indexfile = caclulate_filesize(graphAdjListAndFP_file.c_str());

	pIndex = (uint8_t*)malloc(size_indexfile);
	if (NULL == pIndex)
	{
		printf("Error.. Malloc failed for Graph\n");
		return;
	}

	int nRetIndex = mlock(pIndex, sizeof(size_indexfile));
	cout << "mlock ret for Index: " << nRetIndex << endl;
	if (nRetIndex)
		perror("Index File");
	in3.read((char*)pIndex, size_indexfile);
	in3.close();
	// Sanity test to see if the Index file was loaded properly
	unsigned* puNeighbour = NULL; // Very first neighbour
	unsigned* puNeighbour1 = NULL; // Very Last neighbour
	unsigned *puNumNeighbours = NULL;
	unsigned *puNumNeighbours1 = NULL;
	// First neighbour calculation
	puNumNeighbours = (unsigned*)(pIndex+((INDEX_ENTRY_LEN*0)+ (sizeof(datatype_t)*D) ));
	puNeighbour = puNumNeighbours + 1;
	// Last neighbour calculation
	puNumNeighbours1 = (unsigned*)(pIndex + ( (ullIndex_Entry_LEN * (N-1)) + (sizeof(datatype_t)*D) )) ;
	puNeighbour1 = puNumNeighbours1 + (*puNumNeighbours1) ;
	// Print the first and last neighbour in AdjList
	printf("%u \t %u\n", *puNeighbour, *puNeighbour1);



	datatype_t* queriesFP = NULL;

	queriesFP = (datatype_t*) malloc(sizeof(datatype_t)* numQueries * D);	// full floating point coordinates of queries

	if (NULL == queriesFP)
	{
		printf("Error.. Malloc failed for queriesFP");
		return;
	}

	in4.seekg(8);
	in4.read((char*)queriesFP,sizeof(datatype_t)*D*numQueries);
	in4.close();


	// Loading chunk offsets

	unsigned n_chunks = CHUNKS;
	unsigned *chunksOffset = (unsigned*) malloc(sizeof(unsigned) * (n_chunks+1));
	uint64_t numr = n_chunks + 1;
	uint64_t numc = 1;

	load_bin<uint32_t>(chunkOffsets_file, chunksOffset, numr, numc);	//Import the chunkoffset file


	// Loading centroid coordinates
	float* centroid = nullptr;
	load_bin<float>(centroid_file, centroid, numr, numc);				//Import centroid from centroid file

	// GroundTruth loading done later

	// CPU Data Structs
	unsigned *nearestNeighbours = NULL;
	const unsigned long long FPSetCoords_size = D * sizeof(datatype_t);

	datatype_t* FPSetCoordsList = NULL;

	// Note : R+1 is needed because MEDOID is added as additional neighbour in very first neighbour fetching


	// Final set of K NNs for eacy query will be collected here (sent by GPU)
	nearestNeighbours = (unsigned*)malloc(sizeof(unsigned) * recall_at * numQueries);
	// Allocate host pinned memory for async memcpy
	// [dataset Dimensions * numQuereis] * numIterations
	gpuErrchk(cudaMallocHost(&FPSetCoordsList, (MAX_PARENTS_PERQUERY * numQueries) * FPSetCoords_size));


	// GPU Data Structs
	float *d_pqTable = NULL;
	float *d_pqDistTables = NULL;
	float *d_BestLSetsDist = NULL;
	float *d_neighborsDist_query = NULL;
	float *d_mergedDist = NULL;
	float *d_neighborsDist_query_aux = NULL;
	datatype_t *d_queriesFP = NULL;
	float *d_centroid = NULL;
	unsigned *d_BestLSets = NULL;
	unsigned *d_neighbors = NULL;
	unsigned *d_neighbors_aux = NULL;
	unsigned *d_parents = NULL;
	unsigned *d_mergedNodes = NULL;
	unsigned *d_numNeighbors_query = NULL;
	unsigned *d_BestLSets_count = NULL;
	unsigned *d_iter = NULL;
	unsigned *d_neighbors_temp = NULL;
	unsigned *d_numNeighbors_query_temp = NULL;
	unsigned *d_recall = NULL;
	unsigned *d_mark = NULL;
	uint8_t *d_pIndex;

	unsigned  *d_chunksOffset = NULL;
	uint8_t * d_compressedVectors = NULL;
	bool *d_BestLSets_visited = NULL;
	bool *d_merged_visited = NULL;
	bool *d_nextIter = NULL;
	// ToDo : Check if we can avoid work at bool granularity and at bit level
	bool *d_processed_bit_vec = NULL;
	unsigned* d_numQueries = NULL;
	// 2D array format
	// [FP for Q1] [FP for Q2]...[FP for QN]
	// [FP for Q1] [FP for Q2]...[FP for QN]
	// ..
	// [FP for Q1] [FP for Q2]...[FP for QN] total M such entries (rows)
	// Every iteration: [1 * numQuereis] row added
	// Dimensoins of 2D array : [numIterations * numQueries]
	// numIterations upper bound is MAX_PARENTS_PERQUERY
	datatype_t* d_FPSetCoordsList = NULL; // M x N
	// Indicates how many entries are present per query

	unsigned* d_FPSetCoordsList_Counts = NULL; // Size is N
	float* d_L2distances = NULL; // M x N dimensions
	unsigned* d_L2ParentIds = NULL; // // M x N dimensions

	float* d_L2distances_aux = NULL; // M x N dimensions
	unsigned* d_L2ParentIds_aux = NULL; // // M x N dimensions


	unsigned *d_nearestNeighbours = NULL;
	// A specific stream to to H2D of the FP vectors
 	cudaStream_t streamFPTransfers;
 	cudaStream_t streamKernels;

 	cudaStream_t streamParent;
 	cudaStream_t streamChildren;

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
	unsigned numThreads_K3_merge = 2*L;
	assert(numThreads_K3_merge <= 1024);	// Max thread block size
	unsigned numThreads_K3 = R+1;
	time_B2_vec.push_back(0.0);
	bool nextIter = false;
	unsigned iter = 1; // Note 1-based not 0


	vector<vector<unsigned>> final_bestL1;	// Per query vector to store the visited parent and its distance to query point
	final_bestL1.resize(numQueries);


	// Allocations on GPU
	//gpuErrchk(cudaMalloc(&d_compressedVectors, sizeof(uint8_t) * N * CHUNKS)); 	//100M*100 ~10GB
	gpuErrchk(cudaMalloc(&d_processed_bit_vec, sizeof(bool)*BF_MEMORY*numQueries));
	gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));
	gpuErrchk(cudaMalloc(&d_iter, sizeof(unsigned)));
	gpuErrchk(cudaMemset(d_iter,0,sizeof(unsigned)));
	gpuErrchk(cudaMalloc(&d_pqTable, sizeof(float) * (256*D)));
	gpuErrchk(cudaMalloc(&d_pqDistTables, sizeof(float) * (256*CHUNKS*numQueries)));
	gpuErrchk(cudaMalloc(&d_mergedDist, sizeof(float) * (numQueries* (2*L))));
	gpuErrchk(cudaMalloc(&d_queriesFP, sizeof(datatype_t) * (numQueries*D)));
	gpuErrchk(cudaMalloc(&d_mergedNodes, sizeof(unsigned) * (2*L)));
	gpuErrchk(cudaMalloc(&d_BestLSets, sizeof(unsigned) * (numQueries* (L))));
	gpuErrchk(cudaMalloc(&d_BestLSets_visited, sizeof(bool) * (numQueries* (L))));
	gpuErrchk(cudaMalloc(&d_merged_visited, sizeof(bool) * (numQueries* (2*L))));
	gpuErrchk(cudaMalloc(&d_BestLSetsDist, sizeof(float) * (numQueries*(L))));
	gpuErrchk(cudaMalloc(&d_neighbors_aux, sizeof(unsigned) * (numQueries*(R+1))));
	gpuErrchk(cudaMalloc(&d_numNeighbors_query, sizeof(unsigned) * (numQueries)));

	gpuErrchk(cudaMalloc(&d_neighborsDist_query, sizeof(float) * (numQueries*(R+1))));
	gpuErrchk(cudaMalloc(&d_neighborsDist_query_aux, sizeof(float) * (numQueries*(R+1))));
	gpuErrchk(cudaMalloc(&d_chunksOffset, sizeof(unsigned) * (n_chunks+1)));

	gpuErrchk(cudaMalloc(&d_parents, sizeof(unsigned) * (numQueries*(SIZEPARENTLIST))));
	gpuErrchk(cudaMalloc(&d_neighbors, sizeof(unsigned) * (numQueries*(R+1))));
	gpuErrchk(cudaMalloc(&d_neighbors_temp, sizeof(unsigned) * (numQueries*(R+1))));
	gpuErrchk(cudaMalloc(&d_numNeighbors_query_temp, sizeof(unsigned) * (numQueries)));


	gpuErrchk(cudaMalloc(&d_BestLSets_count, sizeof(unsigned) * (numQueries)));
	gpuErrchk(cudaMalloc(&d_mark, sizeof(unsigned) * (numQueries)));			// ~40KB
	gpuErrchk(cudaMalloc(&d_centroid, sizeof(float) * (D)));			//4*128 ~512B

	gpuErrchk(cudaMalloc(&d_FPSetCoordsList, (MAX_PARENTS_PERQUERY * numQueries) * FPSetCoords_size )); // Dim: [numIterations * numQueries]
	gpuErrchk(cudaMalloc(&d_FPSetCoordsList_Counts, numQueries * sizeof(unsigned) ));
	gpuErrchk(cudaMalloc(&d_L2distances, (MAX_PARENTS_PERQUERY * numQueries) * sizeof(float) )); // Dim: [numIterations * numQueries]
	gpuErrchk(cudaMalloc(&d_L2ParentIds, (MAX_PARENTS_PERQUERY * numQueries) * sizeof(unsigned) )); // Dim: [numIterations * numQueries]
	gpuErrchk(cudaMalloc(&d_nearestNeighbours, (recall_at * numQueries) * sizeof(unsigned) ));// Dim: [recall_at * numQueries]
	gpuErrchk(cudaMalloc(&d_numQueries,sizeof(unsigned)));

	gpuErrchk(cudaMalloc(&d_L2distances_aux, (MAX_PARENTS_PERQUERY * numQueries) * sizeof(float) )); // Dim: [numIterations * numQueries]
	gpuErrchk(cudaMalloc(&d_L2ParentIds_aux, (MAX_PARENTS_PERQUERY * numQueries) * sizeof(unsigned) )); // Dim: [numIterations * numQueries]
	gpuErrchk(cudaMalloc(&d_recall,sizeof(unsigned)));
	gpuErrchk(cudaMemcpy(d_recall, &recall_at, sizeof(unsigned), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_numQueries, &numQueries, sizeof(unsigned), cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy(d_centroid, centroid, sizeof(float) * (D), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_pIndex, size_indexfile));

	// Default stream computations or transfers cannot be overalapped with operations on other streams
	// Hence creating separate streams for transfers and computations to achieve overlap
	// memory transfers overlap with all kernel executions
	gpuErrchk(cudaStreamCreate(&streamFPTransfers));
	gpuErrchk(cudaStreamCreate(&streamKernels));

	gpuErrchk(cudaStreamCreate(&streamParent));
	gpuErrchk(cudaStreamCreate(&streamChildren));

	GPUTimer gputimer (streamKernels,!bEnableGPUStats);	// Initiating the GPUTimer class object

	// transpose pqTable
	float *pqTable_T = NULL; // always float irrespective of datatype_t

	pqTable_T = (float*) malloc(sizeof(float) * (256 * D));
	for(unsigned row = 0; row < 256; ++row) {
		for(unsigned col = 0; col < D; ++col) {
			pqTable_T[col* 256 + row] = pqTable[row*D+col];
		}
	}

	// host to device transfer
	gpuErrchk(cudaMemcpy(d_pqTable, pqTable_T, sizeof(float) * (256*D), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(d_compressedVectors, compressedVectors, (unsigned long long)(sizeof(uint8_t) * (unsigned long long)(CHUNKS)*N),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_chunksOffset, chunksOffset, sizeof(unsigned) * (n_chunks+1), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_pIndex, pIndex, size_indexfile, cudaMemcpyHostToDevice));

	// There are 3 stages for free'ing: 1) After transferring to Device (i.e. before search) 2) After the iterations and 3) Before termination
#ifdef FREE_AFTERUSE
	// ToDo : To reduce CPU peak memory, the compressed vectors cna be transferred to GPU first and free'd. Then load the graph on CPU
	free(compressedVectors);
	compressedVectors = NULL;
	free(pqTable_T);
	pqTable_T = NULL;
	free(chunksOffset);
	chunksOffset = NULL;
	sleep(10); // waiting to get the free(compressedVectors) to settle down
#endif

	// Transfer the Medoid (seed parent) default first parent
	unsigned* L2ParentIds = (unsigned*)malloc(sizeof(unsigned) * numQueries);
	unsigned* FPSetCoordsList_Counts = (unsigned*)malloc(sizeof(unsigned) * numQueries);

do // this is just to run the entire search multiple runs for consistent stats reporting
{
	for (int i = 0 ; i < numQueries; i++)
	{
		L2ParentIds[i] = medoidID;
		FPSetCoordsList_Counts[i] = 1;
	}

	gpuErrchk(cudaMemset(d_pqDistTables,0,sizeof(float) * (CHUNKS * 256 * numQueries)));
	gpuErrchk(cudaMemset(d_processed_bit_vec, 0, sizeof(bool)*BF_MEMORY*numQueries));
	gpuErrchk(cudaMemset(d_parents, 1, sizeof(unsigned)*(numQueries*(SIZEPARENTLIST))));
	gpuErrchk(cudaMemset(d_BestLSets_count, 0, sizeof(unsigned)*(numQueries)));
	gpuErrchk(cudaMemset(d_mark, 1, sizeof(unsigned)*(numQueries)));
	// Note: parent for row 0, should be medoidID, but initializing entirely to medoidID just hurt
	gpuErrchk(cudaMemcpy(d_L2ParentIds, L2ParentIds, sizeof(unsigned) * numQueries, cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy(d_FPSetCoordsList_Counts, FPSetCoordsList_Counts, sizeof(unsigned) * numQueries, cudaMemcpyHostToDevice ));

#ifdef FREE_AFTERUSE
	free(L2ParentIds);
	L2ParentIds = NULL;
	free(FPSetCoordsList_Counts);
	FPSetCoordsList_Counts = NULL;
#endif

	// Cache Warm-up
	if (bCacheWarmUp)
	{
		// CPU  warm-up
		NodeIDMap mapNodeIDToNode;
		unsigned visit_counter = 0;
		SetupBFS(mapNodeIDToNode);
		bfs(medoidID,1000000,visit_counter,mapNodeIDToNode, pIndex);
		ExitBFS(mapNodeIDToNode);

		// GPU warm-up
		unsigned *d_neighbors_warmup = NULL;
		unsigned uNeighbours_size = mapNodeIDToNode.size()/R;
		unsigned *neighbors_warmup = (unsigned*)malloc(sizeof(unsigned) * uNeighbours_size);
		gpuErrchk(cudaMalloc(&d_neighbors_warmup, sizeof(unsigned) * uNeighbours_size));
		gpuErrchk(cudaMemcpy(d_neighbors_warmup, neighbors_warmup, sizeof(unsigned) * uNeighbours_size, cudaMemcpyHostToDevice));
		compute_neighborDist_par_cachewarmup<<< uNeighbours_size, uNeighbours_size/R >>>(d_neighbors_warmup,d_compressedVectors);

	}

	// Cache Warm-up

	omp_set_num_threads(numCPUthreads);

	auto nanoStart = log_message("SEARCH STARTED");
	auto start = std::chrono::high_resolution_clock::now();
	gpuErrchk(cudaMemcpy(d_queriesFP, queriesFP, sizeof(datatype_t) * (D*numQueries), cudaMemcpyHostToDevice));
	auto stop = std::chrono::high_resolution_clock::now();
	time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;
	
	gputimer.Start();
	gpuErrchk(cudaMemset(d_numNeighbors_query, 0, sizeof(unsigned)*numQueries));
	/** [4] Launching the kernel with "numQueries" number of thread-blocks and block size of 256
	 * One thread block is assigned to a query, i.e., 256 threads perform the computation for a query. The block size has been tuned for performance.
	 */
	 // ToDo: 256 or R
	neighbor_filtering_new<<<numQueries, numThreads_K5, 0, streamKernels >>> (d_neighbors, d_neighbors_temp, d_numNeighbors_query, d_numNeighbors_query_temp, d_processed_bit_vec, d_parents, d_pIndex, iter, d_nextIter);
	gputimer.Stop();
	time_neighbor_filtering += gputimer.Elapsed() ;
	unsigned int bang_counter = 0;
	unsigned int* bang_counter_tmp = (unsigned*)malloc(sizeof(unsigned) * numQueries);
	gpuErrchk(cudaMemcpy(bang_counter_tmp, d_numNeighbors_query, sizeof(unsigned) * numQueries, cudaMemcpyDeviceToHost ));

	for (int i=0; i< numQueries; i++)
	{
		bang_counter += bang_counter_tmp[i];
	}

	gputimer.Start();

	/** [5] Launching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K2" block size.
	 * One thread block is assigned to a query, i.e., "numThreads_K2" threads perform the computation for a query.
	 */

	compute_neighborDist_par <<<numQueries, numThreads_K2,0, streamKernels >>> (d_neighbors, d_numNeighbors_query, d_neighborsDist_query, d_queriesFP, d_pIndex);
	// [7] Compute distance of MEDOID to Query Points
	gputimer.Stop();
	time_B1_vec.push_back(gputimer.Elapsed());


	gputimer.Start();
	

#ifdef OLD_MERGE
	/** [8] Launching the kernel with "numQueries" number of thread-blocks and (R+1) block size.
	 * One thread block is assigned to a query, i.e., (R+1) threads perform the computation for a query.
	 * The kernel  sorts an array of size (R+1) per query, so we do not require (R+1) threads per query.
	 */
	compute_BestLSets_par_sort_msort<<<numQueries, numThreads_K3,0, streamKernels >>>(d_neighbors,
														d_neighbors_aux,
														d_numNeighbors_query,
														d_neighborsDist_query,
														d_neighborsDist_query_aux,
														d_nextIter);

	/** [9] Launching the kernel with "numQueries" number of thread-blocks and (2*L) block size.
	 * One thread block is assigned to a query, i.e., (2*L) threads perform the computation for a query.
	 * The kernel merges, for every query, two arrays each of whose sizes are upperbounded by L, so we do not require more than 2*L threads per query.
	 */

	compute_BestLSets_par_merge<<<numQueries, numThreads_K3_merge,0, streamKernels >>>(d_neighbors,
									d_numNeighbors_query,
									d_neighborsDist_query,
									d_BestLSets,
									d_BestLSetsDist,
									d_BestLSets_visited,
									d_parents,
									iter,
									d_nextIter,
									d_BestLSets_count,
	 								d_L2ParentIds,
	 								d_FPSetCoordsList_Counts,
	 								d_numQueries);
#else

	compute_BestLSets_par_sort_msort_new<<<numQueries, max(numThreads_K3,numThreads_K3_merge),0, streamKernels >>>(d_neighbors,
															//d_neighbors_aux,
															d_numNeighbors_query,
															d_neighborsDist_query,
															//d_neighborsDist_query_aux,
															d_BestLSets,
															d_BestLSetsDist,
															d_BestLSets_visited,
															d_parents,
															iter,
															d_nextIter,
															d_BestLSets_count,
															d_L2ParentIds,
															d_FPSetCoordsList_Counts,
															d_numQueries);

													

#endif
	gputimer.Stop();
	time_B2_vec.push_back(gputimer.Elapsed());	

	// Loop until all the query have no new parent
	do
	{
		//printf("Start of iter %d\n", iter);
		gputimer.Start();

		++iter;
		gpuErrchk(cudaMemset(d_numNeighbors_query, 0, sizeof(unsigned)*numQueries));
		/** [11] Launching the kernel with "numQueries" number of thread-blocks and block size of 256
		 * One thread block is assigned to a query, i.e., 256 threads perform the computation for a query. The block size has been tuned for performance.
		 */
		neighbor_filtering_new<<<numQueries, numThreads_K5,0, streamKernels >>> (d_neighbors,
																		d_neighbors_temp,
																		d_numNeighbors_query,
																		d_numNeighbors_query_temp,
																		d_processed_bit_vec,
																		d_parents,
																		d_pIndex,
																		iter,
																		d_nextIter);
		gputimer.Stop();
		time_neighbor_filtering += gputimer.Elapsed() ;

		gpuErrchk(cudaMemcpy(bang_counter_tmp, d_numNeighbors_query, sizeof(unsigned) * numQueries, cudaMemcpyDeviceToHost ));
		for (int i=0; i< numQueries; i++)
		{
			bang_counter += bang_counter_tmp[i];
		}

		gputimer.Start();


		compute_neighborDist_par <<<numQueries, numThreads_K2,0, streamKernels >>> (d_neighbors, d_numNeighbors_query, d_neighborsDist_query, d_queriesFP, d_pIndex);

		gputimer.Stop();
		time_B1_vec.push_back(gputimer.Elapsed());
		

		gputimer.Start();

#ifdef OLD_MERGE
		/** [8] Launching the kernel with "numQueries" number of thread-blocks and (R+1) block size.
		 * One thread block is assigned to a query, i.e., (R+1) threads perform the computation for a query.
		 * The kernel  sorts an array of size (R+1) per query, so we do not require (R+1) threads per query.
		 */
		compute_BestLSets_par_sort_msort<<<numQueries, numThreads_K3,0, streamKernels >>>(d_neighbors,
															d_neighbors_aux,
															d_numNeighbors_query,
															d_neighborsDist_query,
															d_neighborsDist_query_aux,
															d_nextIter);

		/** [9] Launching the kernel with "numQueries" number of thread-blocks and (2*L) block size.
		 * One thread block is assigned to a query, i.e., (2*L) threads perform the computation for a query.
		 * The kernel merges, for every query, two arrays each of whose sizes are upperbounded by L, so we do not require more than 2*L threads per query.
		 */
		compute_BestLSets_par_merge<<<numQueries, numThreads_K3_merge,0, streamKernels >>>(d_neighbors,
										d_numNeighbors_query,
										d_neighborsDist_query,
										d_BestLSets,
										d_BestLSetsDist,
										d_BestLSets_visited,
										d_parents,
										iter,
										d_nextIter,
										d_BestLSets_count,
		 								d_L2ParentIds,
		 								d_FPSetCoordsList_Counts,
		 								d_numQueries);
#else
		compute_BestLSets_par_sort_msort_new<<<numQueries, max(numThreads_K3,numThreads_K3_merge),0, streamKernels >>>(d_neighbors,
															//d_neighbors_aux,
															d_numNeighbors_query,
															d_neighborsDist_query,
															//d_neighborsDist_query_aux,
															d_BestLSets,
															d_BestLSetsDist,
															d_BestLSets_visited,
															d_parents,
															iter,
															d_nextIter,
															d_BestLSets_count,
															d_L2ParentIds,
															d_FPSetCoordsList_Counts,
															d_numQueries
															);

													
#endif

		gputimer.Stop();
		time_B2_vec.push_back(gputimer.Elapsed());

		//printf("here\n%d\n",nextIter);
		start = std::chrono::high_resolution_clock::now();

		gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));  //d_nextIter calculated in compute_parent<<< >>>
		// Note: Default Stream operations (cmputation or memory transfers) cannot overlap with operatiosn on other sterrams.
		// Hence, the above call could act as a synchronization mechanism to ensure all kernels are done (next parent ready)
		// before we start seeking neighbours on CPU

	        // printf("Iteration = %d\n", iter);


		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

		//#ifdef _DBG
		if (iter == MAX_PARENTS_PERQUERY-1)
		{
			printf("Error: Iterations crossed the assumed limit. FPSetCoords size oversun \n");
			break;
		}
		//#endif

		//printf("end of iter %d\n", iter);
	}
	while(nextIter);

#ifdef _DBG1
	fileParents.close();
#endif


#ifdef FREE_AFTERUSE
	// Free memory that is not used after iterations. This helps to reduce the peak mem usage
	// This can negatively impact search performance
	// GPU
	gpuErrchk(cudaFree(d_compressedVectors));
	// ToDo : More data structs can be free'd
	gpuErrchk(cudaFree(d_chunksOffset));
	gpuErrchk(cudaFree(d_pqTable));
	gpuErrchk(cudaFree(d_pqDistTables));
	gpuErrchk(cudaFree(d_processed_bit_vec));
	gpuErrchk(cudaFree(d_nextIter));
	gpuErrchk(cudaFree(d_iter));
	gpuErrchk(cudaFree(d_mergedDist));
	gpuErrchk(cudaFree(d_mergedNodes));
	gpuErrchk(cudaFree(d_BestLSets));
	gpuErrchk(cudaFree(d_BestLSets_visited));
	gpuErrchk(cudaFree(d_merged_visited));
	gpuErrchk(cudaFree(d_BestLSetsDist));
	gpuErrchk(cudaFree(d_neighbors));
	gpuErrchk(cudaFree(d_neighbors_aux));
	gpuErrchk(cudaFree(d_numNeighbors_query));
	gpuErrchk(cudaFree(d_neighborsDist_query));
	gpuErrchk(cudaFree(d_neighborsDist_query_aux));
	gpuErrchk(cudaFree(d_parents));
	gpuErrchk(cudaFree(d_BestLSets_count));
	gpuErrchk(cudaFree(d_neighbors_temp));
	gpuErrchk(cudaFree(d_numNeighbors_query_temp));
	gpuErrchk(cudaFree(d_mark));
	gpuErrchk(cudaFree(d_centroid));


	nRetIndex = munlock(pIndex, size_indexfile);
	if (nRetIndex)
		perror("Index File");

	free(pIndex);
	pIndex = NULL;

#endif // #if FREE_AFTERUSE


//	gputimer.Stop();
//	time_transfer += gputimer.Elapsed();

	// re-rnking start

	gputimer.Start();
#if NOTNECESSARY	
	cudaStreamSynchronize(streamFPTransfers);
	compute_L2Dist<<<numQueries, K4_blockSize >>> (d_FPSetCoordsList,
												d_FPSetCoordsList_Counts,
												d_queriesFP,
												d_L2ParentIds,
												d_L2distances,
												d_nearestNeighbours,
												d_numQueries,
												d_pIndex);

#ifdef _DBG1
	cudaDeviceSynchronize();
	unsigned* pTempDists = (unsigned*)malloc(sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempDists, d_L2distances, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	unsigned* pTempParents = (unsigned*)malloc(sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempParents, d_L2ParentIds, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	unsigned* pTempNumParents = (unsigned*)malloc(sizeof(unsigned) * numQueries);
	gpuErrchk(cudaMemcpy(pTempNumParents, d_FPSetCoordsList_Counts, sizeof(unsigned) * numQueries,
	cudaMemcpyDeviceToHost));

	// print parentIDs and distances for Query 0

	for (int i=0; i< pTempNumParents[nQueryID]; i++ )
	{
		printf("Parent = %d \t distance = %d\n", pTempParents[(numQueries*i) + nQueryID ], pTempDists[(numQueries*i) + nQueryID ] );
	}

#endif
#endif
	compute_NearestNeighbours<<<numQueries, MAX_PARENTS_PERQUERY >>> (d_L2ParentIds,
												d_L2ParentIds_aux,
												d_FPSetCoordsList_Counts,
												d_L2distances,
												d_L2distances_aux,
												d_nearestNeighbours,
												d_numQueries,
												d_recall,
												d_BestLSets);



#ifdef _DBG1
	cudaDeviceSynchronize();
	pTempDists = (unsigned*)malloc(sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempDists, d_L2distances, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	pTempParents = (unsigned*)malloc(sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempParents, d_L2ParentIds, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	pTempNumParents = (unsigned*)malloc(sizeof(unsigned) * numQueries);
	gpuErrchk(cudaMemcpy(pTempNumParents, d_FPSetCoordsList_Counts, sizeof(unsigned) * numQueries,
	cudaMemcpyDeviceToHost));

	// print parentIDs and distances for Query

	printf ("After Sorting\n");
	for (int i=0; i< pTempNumParents[nQueryID]; i++ )
	{
		printf("nQueryID = %d Parent = %d \t distance = %d\n", nQueryID, pTempParents[(numQueries*i) + nQueryID ], pTempDists[(numQueries*i) + nQueryID ] );
	}


#endif
	gputimer.Stop();


	fp_set_time_gpu += gputimer.Elapsed() ;

	start = std::chrono::high_resolution_clock::now();
	gpuErrchk(cudaMemcpy(nearestNeighbours, d_nearestNeighbours, sizeof(unsigned) * (recall_at * numQueries),
				cudaMemcpyDeviceToHost));
	stop = std::chrono::high_resolution_clock::now();
	time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

	// re-rnking end

	auto nanoEnd = log_message("SEARCH END");
	omp_set_num_threads(numCPUthreads);
	#pragma omp parallel
	{
		int CPUthreadno =  omp_get_thread_num();
		vector<unsigned> query_best;

		for(unsigned ii=CPUthreadno; ii < numQueries; ii = ii + numCPUthreads)
		{
			query_best.clear();
			for(unsigned jj=0; jj < recall_at; ++jj)
			{
				query_best.push_back(nearestNeighbours[ ( numQueries * jj) + ii]);
			}

			#pragma omp critical
			{
				final_bestL1[ii] = query_best;
//				printf("final_bestL1[%d] size = %lu \n", ii, final_bestL1[ii].size());
			}
		}
	}



#ifdef FREE_AFTERUSE
	free(queriesFP);
	queriesFP = NULL;
	gpuErrchk(cudaFreeHost(FPSetCoordsList));
	free(nearestNeighbours);
	nearestNeighbours = NULL;

	free(bang_counter_tmp);
	gpuErrchk(cudaFree(d_queriesFP));
	gpuErrchk(cudaFree(d_FPSetCoordsList));
	gpuErrchk(cudaFree(d_FPSetCoordsList_Counts));
	gpuErrchk(cudaFree(d_L2distances));
	gpuErrchk(cudaFree(d_L2ParentIds));
	gpuErrchk(cudaFree(d_nearestNeighbours));
	gpuErrchk(cudaFree(d_recall));
	gpuErrchk(cudaFree(d_numQueries));
	gpuErrchk(cudaFree(d_pIndex));

	cudaStreamDestroy(streamFPTransfers);
	cudaStreamDestroy(streamKernels);

#endif // #if FREE_AFTERUSE

	cout << "iterations = " <<  iter << endl;

	assert(time_B1_vec.size() >= 1);
	float time_B1_avg = time_B1_vec[0];
	time_B1 = time_B1_avg ;
 	for(unsigned idx=1; idx<time_B1_vec.size(); ++idx) {
		time_B1_avg = time_B1_avg + (time_B1_vec[idx] - time_B1_avg)/(idx+1); // running average
		time_B1 += time_B1_vec[idx];
	}

	assert(time_B2_vec.size() >= 1);
	float time_B2_avg = time_B2_vec[0];
	time_B2 = time_B2_avg;
	for(unsigned idx=1; idx<time_B2_vec.size(); ++idx) {
		time_B2_avg = time_B2_avg + (time_B2_vec[idx] - time_B2_avg)/(idx+1); // running average
		time_B2 += time_B2_vec[idx];
	}

	cout << "STATS" << "LINE = " << endl;
	cout << "(1) total time_K1 = " << time_K1 << " ms" << endl;
	cout << "(2) avg. time_B1 = " << time_B1_avg << " ms" << endl;
	cout << "(3) total time_B1 = " << time_B1 << " ms" << endl;;
	cout << "(4) avg. time_B2 = " << time_B2_avg << " ms" << endl;
	cout << "(5) total time_B2 = " << time_B2 << " ms" << endl;
	cout << "(6) total neighbor_filtering_time = " << time_neighbor_filtering  << " ms" << endl;
	cout << "(7) total transfer_time (CPU <--> GPU) = " << time_transfer / 1000 << " ms" << endl;
	cout << "(8) total neigbbour seek time = " << seek_neighbours_time /  1000 << " ms" << endl;
	cout << "(9) Time elapsed in L2 Dist computation (GPU)= " << fp_set_time_gpu  << " ms" << endl;


	double totalTime = time_K1 + time_B1 + time_B2 + time_neighbor_filtering + (time_transfer / 1000) + (seek_neighbours_time / 1000)    ; // in ms
	totalTime += fp_set_time_gpu  ;
	double totalTime_wallclock = (nanoEnd - nanoStart)/1000.0;
	double throughput = (numQueries * 1000.0 * 1000.0) / totalTime_wallclock;
	// Note : (5) not included, becasue it is shadowed by (8)
	cout << "Total time = (1) + (3) + (5) + (6) + (7) + (8) + (9) = " << totalTime << " ms" << endl;
	cout << "Wall Clock Time = " << totalTime_wallclock << " microsec"<< endl;
	cout << "Throughput = " << throughput << " QPS" << endl;
	cout << "Throughput (Exclude Mem Transfers) = " << (numQueries * 1000.0) / ((totalTime_wallclock/1000.0) - time_transfer / 1000.0) << " QPS" << endl;

	// Computing the recall

	unsigned*         gt_ids = nullptr;
	float*            gt_dists = nullptr;
	size_t            gt_num, gt_dim;
	std::vector<uint64_t> Lvec;
	bool calc_recall_flag = false;


	uint64_t curL = L;
	if (curL >= recall_at)
		Lvec.push_back(curL);

	// Like DiskANN o/p we wannted to run with multiple L's in a single invocation of the program
	// So, test_id to each run with a specific L value. But, currently, we have only one L value
	// i.e. Lvec.size() == 1
	if (Lvec.size() == 0) {
		std::cout << "No valid Lsearch found. Lsearch must be at least recall_at."
			<< std::endl;
		exit(1);
	}
	if (file_exists(truthset_bin)) {
		load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim);
		calc_recall_flag = true;
	}
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(2);
	std::string         recall_string = "Recall@" + std::to_string(recall_at);
	cout << "Ls\t" << recall_string  << endl;
	std::vector<std::vector<unsigned> > query_result_ids(Lvec.size());

	unsigned total_size=0;

#ifdef _DBG2
	unsigned int *  FPSetCoordsList_Counts_tmp = (unsigned*)malloc(sizeof(unsigned) * numQueries);

	gpuErrchk(cudaMemcpy(FPSetCoordsList_Counts_tmp, d_FPSetCoordsList_Counts, sizeof(unsigned) * numQueries, cudaMemcpyDeviceToHost ));
	for (int i=0; i< numQueries; i++)
	{
		total_size += FPSetCoordsList_Counts_tmp[i];
	}
	free(FPSetCoordsList_Counts_tmp);
#endif
	for (unsigned test_id = 0; test_id < Lvec.size(); test_id++) {
		unsigned L1 = Lvec[test_id];
		query_result_ids[test_id].resize(recall_at * numQueries);

		for(unsigned ii = 0; ii < numQueries; ++ii) {
			//total_size += final_bestL1[ii].size();
			for(unsigned jj = 0; jj < recall_at; ++jj) {
				query_result_ids[test_id][ii*recall_at+jj] = final_bestL1[ii][jj];
			}
		}
		float recall = 0;
		if (calc_recall_flag)
			recall = calculate_recall(numQueries, gt_ids, gt_dists, gt_dim, query_result_ids[test_id].data(), recall_at, recall_at);
		cout << L1 << "\t" << recall << "\t total parents " << total_size << "\t total computations: " <<
		bang_counter <<  endl;
	}

	// reset counters for next run
	time_K1 = 0.0f;
	time_B1 = 0.0f;
	time_B2 = 0.0f;
	time_B1_vec.clear();
	time_B2_vec.clear();
	time_B1_avg = 0.0f;
	time_B2_avg = 0.0f;
	time_neighbor_filtering = 0.0f;
	time_transfer = 0.0f;
	seek_neighbours_time = 0.0f;
	fp_set_time_gpu = 0.0f;
	iter = 1;

	char c = 'n';
	cout << "Try Next run ? [y|n]" << endl;
	cin >> c;
	if (c !='y')
		break;

}// outer do-while for running the search multiple times. The stats across runs can be averaged etc
// for a reporting results in a consistent manner (i.e. avoid affect of outliers)
while (1);
}




/** This kernel filters out those nodes from the neighborhood of nodes in "d_parents" that  have already been processed before (in a previous iteration) and populates d_neighbors with only the new unseen ones.
 * @param d_neighbors This is populated by the kernel to store the the node ids that will be considered in the next iteration, for each query.
 * @param d_neighbors_temp This stores the node ids of the neighborhood of the nodes fetched from host, for every query.
 * @param d_numNeighbors_query This is populated by the kernel to store the number of nodes in d_neighbors, for every query.
 * @param d_numNeighbors_query_temp This stores the number of neighbors fetched from the host, for every query.
 * @param d_processed_bit_vec This is a boolean vector for keeping track of whether a node is processed on not.
 * @param beamWidth This is the beamwidth.
 */

__global__ void neighbor_filtering_new (unsigned* d_neighbors,
										unsigned* d_neighbors_temp,
										unsigned* d_numNeighbors_query,
										unsigned* d_numNeighbors_query_temp,
										bool* d_processed_bit_vec,
										unsigned* d_parents,
										uint8_t* d_pIndex,
										unsigned iter,
										bool* d_nextIter) {


	unsigned queryID = blockIdx.x;
	unsigned tid = threadIdx.x;

	if(d_parents[queryID*(SIZEPARENTLIST)]==0)	// If number of parent is zero, return.
		return;

	*d_nextIter = false;

	unsigned offset_neighbors = queryID * (R+1); //Offset into d_neighbors_temp array
	unsigned offset_bit_vec = queryID*BF_MEMORY;	//Offset into d_processed_bit_vec vector of bloom filter
	bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;
	unsigned long long parentID;

	if(iter==1){
		// If this is first iteration, set the bits corresponding to MEDOID so that its not taken as parent in next iteration
		parentID = MEDOID;
		if(tid==0){
			if(!((d_processed_bit_vec_start[hashFn1_d(MEDOID)]) 
			//&& 
			//(d_processed_bit_vec_start[hashFn2_d(MEDOID)])
			)
			) {
				d_processed_bit_vec_start[hashFn1_d(MEDOID)] = true;
				//d_processed_bit_vec_start[hashFn2_d(MEDOID)] = true;
				unsigned old = atomicAdd(&d_numNeighbors_query[queryID], 1);
				d_neighbors[offset_neighbors + old] = MEDOID;
			}
		}
	}

	else parentID = d_parents[queryID*(SIZEPARENTLIST)+1];

	unsigned* bound = (unsigned*)(d_pIndex + ((unsigned long long)INDEX_ENTRY_LEN*parentID) + D*sizeof(datatype_t));
	// For each neighbor of current parent in d_graph array check if its corresponding bits in the d_processed_bit_vec are already set
	for(unsigned ii=tid; ii < *bound; ii += blockDim.x ) {
		unsigned nbr = *(bound+1+ii);
		if(!((d_processed_bit_vec_start[hashFn1_d(nbr)]) 
		//&& 
		//(d_processed_bit_vec_start[hashFn2_d(nbr)])
		)) {
			d_processed_bit_vec_start[hashFn1_d(nbr)] = true;
			//d_processed_bit_vec_start[hashFn2_d(nbr)] = true;
			unsigned old = atomicAdd(&d_numNeighbors_query[queryID], 1);
			d_neighbors[offset_neighbors + old] = nbr;
		}
	}
	//__syncthreads();


}

/*Hash functions used for bloom filters */
__device__ unsigned hashFn1_d(unsigned x) {

	// FNV-1a hash
	uint64_t hash = 0xcbf29ce4;
	hash = (hash ^ (x & 0xff)) * 0x01000193 ;
	hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;

	return hash % (BF_ENTRIES );
}

__device__ unsigned hashFn2_d(unsigned x) {

	// FNV-1a hash
	uint64_t hash = 0x84222325;
	hash = (hash ^ (x & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 8) & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 16) & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 24) & 0xff)) * 0x1B3;
	return hash % (BF_ENTRIES);
}


/**
 * This kernel computes distance of the neighbour from the medoid using the compressed vectors
 * THe main intent to is to access the compressed vectors of the neighbour and not the distance calculation (which is dummy operation) as such
 ^ One thread is responsible for distance calcuation of one node
 */


__global__ void  compute_neighborDist_par_cachewarmup(unsigned* d_neighbors,
											uint8_t* d_compressedVectors) {

	unsigned tid = threadIdx.x;
	unsigned queryID = blockIdx.x;
	unsigned numNeighbors = R;
	unsigned queryNeighbors_offset  = queryID * (blockDim.x);	// offset into d_neighbors array
	unsigned* d_neighbors_start = d_neighbors + queryNeighbors_offset;

	for( unsigned j = tid; j < numNeighbors; j += (blockDim.x) ) { // assign eight threads to a neighbor, within a query

		unsigned long long compressed_vector_offset = ((unsigned long long)d_neighbors_start[j])*CHUNKS;

		float sum = 0.0f;
		d_compressedVectors += compressed_vector_offset;
		for(unsigned long long i = tid%8; i < CHUNKS; i += 8 ){

			sum += d_compressedVectors[i] ;
		}
	}
}


/**
 * This kernel computes for every query, the distance between the neighbors and the query through lookups to the PQ Dist Table and the compressed vectors of the neighbors.
 * @param d_neighbors This is the concatenated list of node ids for all queries whose distances have to be computed with the respective queries.
 * @param d_numNeighbors_query This stores for every query the number of nodes in d_neighbors.
 * @param d_compressedVectors This contains the compressed vectors of all the nodes in the dataset.
 * @param d_pqDistTables This is the concatenated PQ Distance Tables of all the queries.
 * @param d_neighborsDist_query This is populated by the kernel with the distances between the node ids in d_neighbors and the respective query for all queries.
 * @param beamWidth This is the beamwidth.
 */

__global__ void  compute_neighborDist_par(unsigned* d_neighbors,
											unsigned* d_numNeighbors_query,
											float*  d_neighborsDist_query,
											datatype_t* d_queriesFP,
											uint8_t* d_pIndex) {

	unsigned tid = threadIdx.x;
	unsigned queryID = blockIdx.x;
	//__shared__ unsigned shm_neighbours[R+1];

	unsigned numNeighbors = d_numNeighbors_query[queryID];
	unsigned queryNeighbors_start  = queryID * (R+1);	// offset into d_neighbors array
	float* d_neighborsDist_query_start = d_neighborsDist_query + queryNeighbors_start;
	datatype_t* d_queriesFP_start = d_queriesFP+(queryID*D);

/*
	for(unsigned uIter = tid; uIter < numNeighbors; uIter += blockDim.x){
		d_neighborsDist_query_start[ uIter] = 0.0;
		//shm_neighbours[uIter] = d_neighbors[queryNeighbors_start + uIter];
	}*/
	//__syncthreads();		
	
	//	__syncthreads();
	#define THREADS_PER_NEIGHBOR 8	
	typedef cub::WarpReduce<float,THREADS_PER_NEIGHBOR> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage[R];

	//blockDim.x need to be multiple of THREADS_PER_NEIGHBOR
	for( unsigned j = tid/THREADS_PER_NEIGHBOR; j < numNeighbors; j += (blockDim.x)/THREADS_PER_NEIGHBOR ) { // assign eight threads to a neighbor, within a query
		unsigned long long myNeighbor = d_neighbors[ queryNeighbors_start + j]; 
		
		datatype_t* pBase = (datatype_t*)(d_pIndex+(myNeighbor*INDEX_ENTRY_LEN));
		float sum = 0.0f;
		for(unsigned i = tid%THREADS_PER_NEIGHBOR; i < D; i +=  THREADS_PER_NEIGHBOR){
			float diff = (float)(*(pBase+i)) - (float)d_queriesFP_start[i];	// d_quriesFP must contain float
			sum += diff*diff;	// Parallel execution
		}
		d_neighborsDist_query_start[j] = WarpReduce(temp_storage[j]).Sum(sum);
		//atomicAdd(&d_neighborsDist_query[queryNeighbors_start + j], sum);
	}
}



__global__ void compute_L2Dist (datatype_t* d_FPSetCoordsList,
								unsigned* d_FPSetCoordsList_Counts,
								datatype_t* d_queriesFP,
								unsigned* d_L2ParentIds,
								float* d_L2distances,
								unsigned* d_nearestNeighbours,
								unsigned* d_numQueries,
								uint8_t* d_pIndex)
{

	__shared__ datatype_t query_vec[D];
	//datatype_t query_vec[D];
	unsigned queryID = blockIdx.x;
	unsigned numNodes = d_FPSetCoordsList_Counts[queryID];
	unsigned tid = threadIdx.x;
	unsigned gid = queryID * D;
	unsigned numQueries = *d_numQueries;

	for(unsigned ii= tid; ii < D; ii += blockDim.x) {
		query_vec[ii] = d_queriesFP[gid + ii];
	}
	__syncthreads();

	// one thread block computes the distances of all the nodes for a query,
	for(unsigned ii = tid; ii < numNodes; ii += blockDim.x) {
		datatype_t* pBase = (datatype_t*)(d_pIndex + (unsigned long long)d_L2ParentIds[numQueries*ii + queryID]*INDEX_ENTRY_LEN);
		float L2Dist = 0.0;
		for(unsigned jj=0;jj < D; ++jj) {
			float diff = (float)(*(pBase+jj)) - (float)query_vec[jj];
			L2Dist = L2Dist + (diff * diff);
		}
		d_L2distances[( numQueries * ii) + queryID ] = L2Dist;
	}
}

__global__ void  compute_NearestNeighbours(unsigned* d_L2ParentIds,
						unsigned* d_L2ParentIds_aux,
						unsigned* d_FPSetCoordsList_Counts,
						float* d_L2distances,
						float* d_L2distances_aux,
						unsigned* d_nearestNeighbours,
						unsigned* d_numQueries,
						unsigned* d_recall,
						unsigned* d_BestLSets)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
	#if NOTNCESSARY
    unsigned numNeighbors = d_FPSetCoordsList_Counts[queryID];

    if(tid >= numNeighbors || numNeighbors <= 0) return;

    __shared__ unsigned shm_pos[MAX_PARENTS_PERQUERY + 1]; // Or MAX_PARENTS_PERQUERY+1?

	// perform parallel merge sort
	for(unsigned subArraySize=2; subArraySize< 2*numNeighbors; subArraySize *= 2){
		unsigned subArrayID = tid/subArraySize;
		unsigned start = subArrayID * subArraySize;
		unsigned mid = min(start + subArraySize/2, numNeighbors);
		unsigned end = min(start + subArraySize, numNeighbors);

		if(tid >= start && tid < mid){
			unsigned lowerBound = lower_bound_d_ex(&d_L2distances[mid * (*d_numQueries)], 0, end-mid, d_L2distances[(tid * (*d_numQueries)) + queryID],
								*d_numQueries, queryID);
			shm_pos[tid] = lowerBound + tid;	// Position for this element
		}

		if(tid >= mid && tid < end)  {
			unsigned upperBound = upper_bound_d_ex(&d_L2distances[start * (*d_numQueries)], 0, mid-start, d_L2distances[(tid * (*d_numQueries)) + queryID],
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
	#endif
	for(unsigned ii = tid; ii < *d_recall; ii += blockDim.x)
	{
		d_nearestNeighbours[( (*d_numQueries) * ii) + queryID ] = d_BestLSets[( (L) * queryID) + ii ];
	}
}



/** This kernel sorts the neighbors (for every query) in increasing order of their distances to the query, using parallel merge sort
 * The sorted list of neighbors is stored in "d_neighbors" at the end of the kernel.
 * @param d_neighbors This is the concatenated list of node ids for all queries whose distances have to be computed with the respective queries.
 * @param d_neighbors_aux This is an auxiliary array for d_neighbors, used for merge sort.
 * @param d_numNeighbors_query This stores for every query the number of nodes in d_neighbors.
 * @param d_neighborsDist_query This contains the distances between the node ids in d_neighbors and the respective query for all queries.
 * @param d_neighborsDist_query_aux This is an auxiliary array for d_neighborsDist_query, used for merge sort.
 * @param d_nextIter This maintains boolean flags for all queries, to decide if a query is to processed in the next iteration.
 * @param beamWidth This is the beamwidth.
 */
#ifdef OLD_MERGE
__global__ void  compute_BestLSets_par_sort_msort(unsigned* d_neighbors,
													unsigned* d_neighbors_aux1,
													unsigned* d_numNeighbors_query,
													float* d_neighborsDist_query,
													float* d_neighborsDist_query_aux1,
													bool* d_nextIter) {


	unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    *d_nextIter = false;

    if(tid >= numNeighbors || numNeighbors <= 0) return;

    __shared__ unsigned shm_pos[R+1];
	__shared__ unsigned shm_neighbors_aux[R+1];
	__shared__ unsigned shm_neighborsDist_query_aux[R+1];
    unsigned offset = queryID*(R+1);	// Offset into d_neighborsDist_query, d_neighborsDist_query_aux, d_neighbors_aux and d_neighbors arrays

	// perform parallel merge sort
	for(unsigned subArraySize=2; subArraySize< 2*numNeighbors; subArraySize *= 2){
		unsigned subArrayID = tid/subArraySize;
		unsigned start = subArrayID * subArraySize;
		unsigned mid = min(start + subArraySize/2, numNeighbors);
		unsigned end = min(start + subArraySize, numNeighbors);

		if(tid >= start && tid < mid){
			unsigned lowerBound = lower_bound_d(&d_neighborsDist_query[offset + mid], 0, end-mid, d_neighborsDist_query[offset + tid]);
			shm_pos[tid] = lowerBound + tid;	// Position for this element
		}

		if(tid >= mid && tid < end)  {
			unsigned upperBound = upper_bound_d(&d_neighborsDist_query[offset + start], 0, mid-start, d_neighborsDist_query[offset + tid]);
			shm_pos[tid] = start + (upperBound + tid-mid);	// Position for this element

		}
		__syncthreads();
		__threadfence_block();

		// Copy the neighbors to auxiliary array at their correct position
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			shm_neighborsDist_query_aux[shm_pos[i]] = d_neighborsDist_query[offset+i];
			shm_neighbors_aux[ shm_pos[i]] = d_neighbors[offset+i];
		}
		__syncthreads();
		// Copy the auxiliary array to original array
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			d_neighborsDist_query[offset + i] = shm_neighborsDist_query_aux[i];
			d_neighbors[offset + i] = shm_neighbors_aux[i];
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
			 								unsigned* d_L2ParentIds,
			 								unsigned* d_FPSetCoordsList_Counts,
			 								unsigned* d_numQueries){

	__shared__ float shm_neighborsDist_query[R]; // R+1 is an upperbound on the number of neighbors
	__shared__ float shm_currBestLSetsDist[L];
	__shared__ float shm_BestLSetsDist[L];
	__shared__ unsigned shm_pos[R+L+1];
	__shared__ unsigned shm_BestLSets[L];
	__shared__ bool shm_BestLSets_visited[L];

	unsigned queryID = blockIdx.x;
	unsigned numNeighbors = d_numNeighbors_query[queryID];
	unsigned tid = threadIdx.x;

	unsigned Best_L_Set_size  = 0;
	unsigned newBest_L_Set_size = d_BestLSets_count[queryID];
	unsigned nbrsBound;
	unsigned offset = queryID*(R+1);
	unsigned numQueries = *d_numQueries;


	if(numNeighbors > 0){	// If the number of neighbors after filteration is zero then no sense of merging

        if(iter==1){	// If this is the first call to compute_BestLSets_par_merge by this query then initialize d_BestLSets, d_BestLSetsDist...
                nbrsBound = min(numNeighbors,L);
                for(unsigned ii=tid; ii < nbrsBound; ii += blockDim.x) {
                        unsigned nbr =  d_neighbors[offset + ii];
                        d_BestLSets[queryID*L + tid] = nbr;
                        d_BestLSetsDist[queryID*L + tid] =   d_neighborsDist_query[offset + ii];
                        d_BestLSets_visited[queryID*L + tid] = ( nbr == MEDOID);
                }
                __syncthreads();
                newBest_L_Set_size = nbrsBound;
                d_BestLSets_count[queryID] = nbrsBound;
        }
        else {
                Best_L_Set_size = d_BestLSets_count[queryID];

                float maxBestLSetDist = d_BestLSetsDist[L*queryID+Best_L_Set_size-1];
                for(nbrsBound = 0; nbrsBound < min(L,numNeighbors); ++nbrsBound) {
                        if(d_neighborsDist_query[offset + nbrsBound] >= maxBestLSetDist){
                                break;
                        }
                }


                nbrsBound = max(nbrsBound, min(L-Best_L_Set_size, numNeighbors));	//Added by saim
                // if both Best_L_Set_size and numNeighbors is less than L, then the max of the two will be the newBest_L_Set_size otherwise it will be L
                newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, L);			//Updated by saim

                d_BestLSets_count[queryID] = newBest_L_Set_size;


			/*perform parallel merge */
                for(int i=tid; i < nbrsBound; i += blockDim.x) {
                        shm_neighborsDist_query[i] = d_neighborsDist_query[offset + i];
                }
                for(int i=tid; i < Best_L_Set_size; i += blockDim.x) {
                        shm_currBestLSetsDist[i] = d_BestLSetsDist[L*queryID+i];
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
                        shm_BestLSets[shm_pos[tid]] = d_BestLSets[queryID*L+(tid-nbrsBound)];
                        shm_BestLSets_visited[shm_pos[tid]] = d_BestLSets_visited[queryID*L+(tid-nbrsBound)];
                }
                __syncthreads();
                __threadfence_block();


                //Copying back from shared memory to device array
                if (tid < newBest_L_Set_size) {
                        d_BestLSetsDist[L*queryID+tid] = shm_BestLSetsDist[tid];
                        d_BestLSets[L*queryID+tid] = shm_BestLSets[tid];
                        d_BestLSets_visited[L*queryID+tid] = shm_BestLSets_visited[tid];
                    }
                __syncthreads();
                __threadfence_block();
        }
	}
	if(tid == 0) {
			unsigned parentIndex = 0;
			for(unsigned ii=0; ii < newBest_L_Set_size; ++ii) {
				if(!d_BestLSets_visited[L*queryID + ii]) 
				{
					parentIndex++;
					d_BestLSets_visited[L*queryID + ii] = true;
					d_parents[queryID*(SIZEPARENTLIST)+parentIndex] = d_BestLSets[L*queryID + ii];
					break;
				}
			}
			d_parents[queryID*(SIZEPARENTLIST)] = parentIndex;
			if(parentIndex != 0) // parentIndex == 0 is the termination condition for the algorithm.
				{
					*d_nextIter = true;
					// Note: One thread assigned to one Query, so ok to increment (no contention)
					d_FPSetCoordsList_Counts[queryID]++;
					// ToDo : Ensure to put MEDOID as the first parent
					d_L2ParentIds[(iter * numQueries) + queryID] = d_parents[queryID*(SIZEPARENTLIST)+parentIndex];
				}
	}

}
#else

__global__ void  compute_BestLSets_par_sort_msort_new(unsigned* d_neighbors,
													//unsigned* d_neighbors_aux,
													unsigned* d_numNeighbors_query,
													float* d_neighborsDist_query,
													//float* d_neighborsDist_query_aux,
													unsigned* d_BestLSets,
													float* d_BestLSetsDist,
													bool* d_BestLSets_visited,
													unsigned* d_parents,
													unsigned iter,
													bool* d_nextIter,
													unsigned* d_BestLSets_count,
													unsigned* d_L2ParentIds,
													unsigned* d_FPSetCoordsList_Counts,
													unsigned* d_numQueries													
													) {
	unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    *d_nextIter = false;

    __shared__ unsigned shm_pos[R+1];
    unsigned offset = queryID*(R+1);	// Offset into d_neighborsDist_query, shm_neighborsDist_query_aux, d_neighbors_aux and d_neighbors arrays

	__shared__ float shm_neighborsDist_query_aux[R+1];
	__shared__ unsigned shm_neighbors_aux[R+1];


	__shared__ float shm_neighborsDist_query[R]; // R+1 is an upperbound on the number of neighbors
	__shared__ float shm_currBestLSetsDist[L];
	__shared__ float shm_BestLSetsDist[L];
	__shared__ unsigned shm_pos1[R+L+1];
	__shared__ unsigned shm_BestLSets[L];
	__shared__ bool shm_BestLSets_visited[L];
	__shared__ unsigned Temp;
	
	// perform parallel merge sort
	for(unsigned subArraySize=2; subArraySize< 2*numNeighbors; subArraySize *= 2){
		unsigned subArrayID = tid/subArraySize;
		unsigned start = subArrayID * subArraySize;
		unsigned mid = min(start + subArraySize/2, numNeighbors);
		unsigned end = min(start + subArraySize, numNeighbors);

		if(tid >= start && tid < mid){
			unsigned lowerBound = lower_bound_d(&d_neighborsDist_query[offset + mid], 0, end-mid, d_neighborsDist_query[offset + tid]);
			shm_pos[tid] = lowerBound + tid;	// Position for this element
		}

		if(tid >= mid && tid < end)  {
			unsigned upperBound = upper_bound_d(&d_neighborsDist_query[offset + start], 0, mid-start, d_neighborsDist_query[offset + tid]);
			shm_pos[tid] = start + (upperBound + tid-mid);	// Position for this element

		}
		__syncthreads();
		__threadfence_block();

		// Copy the neighbors to auxiliary array at their correct position
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			shm_neighborsDist_query_aux[ shm_pos[i]] = d_neighborsDist_query[offset+i];
			shm_neighbors_aux[shm_pos[i]] = d_neighbors[offset+i];
		}
		__syncthreads();
		#if 1
		// Copy the auxiliary array to original array
		for(int i=tid; i < numNeighbors; i += blockDim.x) {
			d_neighborsDist_query[offset + i] = shm_neighborsDist_query_aux[i];
			d_neighbors[offset + i] = shm_neighbors_aux[i];
		}
		__syncthreads();
		#endif
	}
	//}
	//{
	__syncthreads();
	#if 1
	for(int i=tid; i < numNeighbors; i += blockDim.x) {
			shm_neighborsDist_query_aux[i] = d_neighborsDist_query[offset + i];
			shm_neighbors_aux[i] = d_neighbors[offset + i];
		}
	__syncthreads();
	//unsigned queryID = blockIdx.x;
	//unsigned numNeighbors = d_numNeighbors_query[queryID];
	//unsigned tid = threadIdx.x;

	 unsigned Best_L_Set_size  ;// = 0;
	 unsigned newBest_L_Set_size ; //= d_BestLSets_count[queryID];
	 __shared__ unsigned nbrsBound;
	//unsigned offset = queryID*(R+1);
	//unsigned numQueries = *d_numQueries;


	if(numNeighbors > 0){	// If the number of neighbors after filteration is zero then no sense of merging

        if(iter==1){	// If this is the first call to compute_BestLSets_par_merge by this query then initialize d_BestLSets, d_BestLSetsDist...
                nbrsBound = min(numNeighbors,L);
                for(unsigned ii=tid; ii < nbrsBound; ii += blockDim.x) {
                        unsigned nbr =  shm_neighbors_aux[ ii];
                        d_BestLSets[queryID*L + tid] = nbr;
                        d_BestLSetsDist[queryID*L + tid] =   shm_neighborsDist_query_aux[ ii];
                        d_BestLSets_visited[queryID*L + tid] = ( nbr == MEDOID);
                }
                __syncthreads();
                newBest_L_Set_size = nbrsBound;
                d_BestLSets_count[queryID] = nbrsBound;
        }
        else {
                Best_L_Set_size = d_BestLSets_count[queryID];

                float maxBestLSetDist = d_BestLSetsDist[L*queryID+Best_L_Set_size-1];
				Temp = min(L,numNeighbors);
				if (tid == 0) {
                for(nbrsBound = 0; nbrsBound < Temp ; ++nbrsBound) {
                        if(shm_neighborsDist_query_aux[ nbrsBound] >= maxBestLSetDist){
                                break;
                        }
                }
				}
				__syncthreads();

                nbrsBound = max(nbrsBound, min(L-Best_L_Set_size, numNeighbors));	//Added by saim
                // if both Best_L_Set_size and numNeighbors is less than L, then the max of the two will be the newBest_L_Set_size otherwise it will be L
                newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, L);			//Updated by saim

                d_BestLSets_count[queryID] = newBest_L_Set_size;


			/*perform parallel merge */
               /* for(int i=tid; i < nbrsBound; i += blockDim.x) {
                        shm_neighborsDist_query[i] = shm_neighborsDist_query_aux[ i];
                }*/
                for(int i=tid; i < Best_L_Set_size; i += blockDim.x) {
                        shm_currBestLSetsDist[i] = d_BestLSetsDist[L*queryID+i];
                }
                __syncthreads();
                if(tid < nbrsBound) {
                        shm_pos1[tid] =  lower_bound_d(shm_currBestLSetsDist, 0, Best_L_Set_size, shm_neighborsDist_query_aux[tid]) + tid;
                }
                if( tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size)) {
                        shm_pos1[tid] =  upper_bound_d(shm_neighborsDist_query_aux, 0, nbrsBound, shm_currBestLSetsDist[tid-nbrsBound]) + (tid-nbrsBound);
                }

                __syncthreads();
                __threadfence_block();

                // all threads of the block have populated the positions array in shared memory
                if(tid < nbrsBound && shm_pos1[tid] < newBest_L_Set_size)  {
                        shm_BestLSetsDist[shm_pos1[tid]] = shm_neighborsDist_query_aux[tid];
                        shm_BestLSets[shm_pos1[tid]] = shm_neighbors_aux[tid];
                        shm_BestLSets_visited[shm_pos1[tid]] = false;
                }
				Temp = (nbrsBound + Best_L_Set_size);
                if(tid >= nbrsBound && tid < (Temp) && shm_pos1[tid] < newBest_L_Set_size) {
                        shm_BestLSetsDist[shm_pos1[tid]] = shm_currBestLSetsDist[tid-nbrsBound];
                        shm_BestLSets[shm_pos1[tid]] = d_BestLSets[queryID*L+(tid-nbrsBound)];
                        shm_BestLSets_visited[shm_pos1[tid]] = d_BestLSets_visited[queryID*L+(tid-nbrsBound)];
                }
                __syncthreads();
                __threadfence_block();

                //Copying back from shared memory to device array
                if (tid < newBest_L_Set_size) {
                        d_BestLSetsDist[L*queryID+tid] = shm_BestLSetsDist[tid];
                        d_BestLSets[L*queryID+tid] = shm_BestLSets[tid];
                        d_BestLSets_visited[L*queryID+tid] = shm_BestLSets_visited[tid];
                    }
                __syncthreads();
                //__threadfence_block();
        }
	}

	if(tid == 0) {
			unsigned parentIndex = 0;
			for(unsigned ii=0; ii < newBest_L_Set_size; ++ii) {
				if(!d_BestLSets_visited[L*queryID + ii]) 
				{
					parentIndex++;
					d_BestLSets_visited[L*queryID + ii] = true;
					d_parents[queryID*(SIZEPARENTLIST)] = parentIndex;
					d_parents[queryID*(SIZEPARENTLIST)+parentIndex] = d_BestLSets[L*queryID + ii];
					*d_nextIter = true;
					break;
				}
			}
			#if 0
			d_parents[queryID*(SIZEPARENTLIST)] = parentIndex;
			if(parentIndex != 0) // parentIndex == 0 is the termination condition for the algorithm.
				{
					*d_nextIter = true;
					// Note: One thread assigned to one Query, so ok to increment (no contention)
					//d_FPSetCoordsList_Counts[queryID]++;
					// ToDo : Ensure to put MEDOID as the first parent
					//d_L2ParentIds[(iter * numQueries) + queryID] = d_parents[queryID*(SIZEPARENTLIST)+parentIndex];
				}
			#endif
	}
#endif
}


#endif

/*Helper device function which returns the position first element in the range [lo,hi), which has a value not less than 'target'.*/
__device__ unsigned lower_bound_d(float arr[], unsigned lo, unsigned hi, float target) {
	unsigned mid;

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
__device__ unsigned upper_bound_d(float arr[], unsigned lo, unsigned hi, float target) {

	unsigned mid;

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
__device__ unsigned lower_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target, unsigned row_size, unsigned queryID)
{
	unsigned mid;

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
__device__ unsigned upper_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target,  unsigned row_size, unsigned queryID)
{

	unsigned mid;

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
NeighbourList GetNeighbours(uint8_t* pGraph,
							unsigned curreParent,
							NodeIDMap& mapNodeIDToNode)
{
	NeighbourList retList;
	// find the children nodes and its degree
	unsigned long long temp = (ullIndex_Entry_LEN * curreParent) + (D*sizeof(datatype_t));
	unsigned *puNumNeighbours = (unsigned*)(pGraph + temp );

	for(unsigned kk = 0; kk < *puNumNeighbours; ++kk)
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

void bfs(unsigned uMedoid,
		const unsigned nNodesToDiscover,
		unsigned& visit_counter,
		NodeIDMap& mapNodeIDToNode,
		uint8_t* pGraph)
{
	Node* pNode = (Node*)malloc(sizeof(Node));
	pNode->uNodeID = uMedoid;
	pNode->bVisited = true;
	//pNode->nLevel = 0;
	mapNodeIDToNode[pNode->uNodeID] = pNode;
	visit_counter++;
	list<unsigned> queue;
	queue.push_back(uMedoid);

	while (!queue.empty())
	{
		bool bRet = false;
		unsigned currentVertex = queue.front();
		queue.pop_front();
		//printf("Visited %d\n", currentVertex);
		NeighbourList listChildres = GetNeighbours(pGraph, currentVertex, mapNodeIDToNode);

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
