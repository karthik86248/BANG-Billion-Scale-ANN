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
#include "parANN.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>


#define R 64 // Max. node degree

#define BF_ENTRIES  399887U    	 // per query, max entries in BF, (prime number)
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned); // 4-byte mem aligned size for actual allocation

#define _DBG_BOUNDS
int nQueryID = 0;

// Indicates MAX iterations performed. IF there is at least one qeuery that requires neighbour seek
// for a parent, then iteration will occur. There is one initial iteratoion where Medoid is added (outside do-while)
#define MAX_PARENTS_PERQUERY (L + 50) // Needs to be set with expereince. set it to (2*L) if in doubt

// length of each entry in INDEX file 128 bytes (FP) + 4 bytes (degree) + 256 bytes (AdjList for R=64)
// There are N such entries
#define SIZEPARENTLIST  (1+1) // one to indicate present/abset and other the actual parent

// Capabilities
#define ENABLE_GPU_STATS 		0x00000001
#define ENABLE_CACHE_WARMUP 	0x00000002

using namespace std;
using Clock = std::chrono::high_resolution_clock;

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
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	struct tm *local = localtime(&ctimenow_obj);
	//printf("Today is %s %s\n", ctime(&ctimenow_obj),message);
	cout <<  local->tm_hour <<":" << local->tm_min <<":"<< local->tm_sec << " [ " << millis << " ] : " << message << endl;
	return millis;
}


void parANN(int argc, char** argv) {
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
	unsigned K4_blockSize = 256;

	cout << medoidID << "\t" << numQueries << endl;

	printf("BF Memory = %u BF_ENTRIES = %u\n",BF_MEMORY, BF_ENTRIES);

	// Check if files exist
	ifstream in1(pqTable_file, std::ios::binary);
	if(!in1.is_open()){
		printf("Error.. Could not open the file1..");
		return;
	}
	ifstream in2(compressedVector_file, std::ios::binary);
	if(!in2.is_open()){
		printf("Error.. Could not open the file2..");
		return;
	}

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
	// To reduce Peak Host memory usage, loading compressed vectors to CPU and transferring to GPU and releasing host memory
	uint8_t* d_compressedVectors = NULL;
	gpuErrchk(cudaMalloc(&d_compressedVectors, sizeof(uint8_t) * N * CHUNKS)); 	//100M*100 ~10GB
	gpuErrchk(cudaMemcpy(d_compressedVectors, compressedVectors, (unsigned long long)(sizeof(uint8_t) * (unsigned long long)(CHUNKS)*N),
	cudaMemcpyHostToDevice));
	free(compressedVectors);
	compressedVectors = NULL;
	printf("Transferring Compressed vectors done\n");
	sleep(10);




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
	unsigned *neighbors = NULL;
	unsigned *numNeighbors_query = NULL;
	unsigned *parents = NULL;
	unsigned *nearestNeighbours = NULL;
	const unsigned long long FPSetCoords_size_bytes = D * sizeof(datatype_t);

	const unsigned long long FPSetCoords_rowsize_bytes = FPSetCoords_size_bytes * numQueries;

	const unsigned long long FPSetCoords_size = D ;

	const unsigned long long FPSetCoords_rowsize = FPSetCoords_size * numQueries;


	datatype_t* FPSetCoordsList = NULL;

	// Note : R+1 is needed because MEDOID is added as additional neighbour in very first neighbour fetching
	gpuErrchk(cudaMallocHost(&neighbors, sizeof(unsigned) * (numQueries*(R+1))) );
	gpuErrchk(cudaMallocHost(&numNeighbors_query,sizeof(unsigned) * (numQueries) ));
	gpuErrchk(cudaMallocHost(&parents,sizeof(unsigned) * numQueries * (SIZEPARENTLIST)));


	// Final set of K NNs for eacy query will be collected here (sent by GPU)
	nearestNeighbours = (unsigned*)malloc(sizeof(unsigned) * recall_at * numQueries);
	// Allocate host pinned memory for async memcpy
	// [dataset Dimensions * numQuereis] * numIterations
	gpuErrchk(cudaMallocHost(&FPSetCoordsList, (MAX_PARENTS_PERQUERY * numQueries) * FPSetCoords_size_bytes));


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

	unsigned  *d_chunksOffset = NULL;
	bool *d_BestLSets_visited = NULL;
	bool *d_nextIter = NULL;
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
	unsigned offset =  (R+1);

	// Allocations on GPU
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

	gpuErrchk(cudaMalloc(&d_FPSetCoordsList, (MAX_PARENTS_PERQUERY * numQueries) * FPSetCoords_size_bytes )); // Dim: [numIterations * numQueries]
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
	gpuErrchk(cudaMemcpy(d_chunksOffset, chunksOffset, sizeof(unsigned) * (n_chunks+1), cudaMemcpyHostToDevice));

	// There are 3 stages for free'ing: 1) After transferring to Device (i.e. before search) 2) After the iterations and 3) Before termination
#ifdef FREE_AFTERUSE
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
	gpuErrchk(cudaMemset(d_parents, 0, sizeof(unsigned)*(numQueries*(SIZEPARENTLIST))));
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
	// [1] collecting all medoid's neighbors for better locality
	puNumNeighbours = (unsigned*)( pIndex + ((ullIndex_Entry_LEN * medoidID) + (D*sizeof(datatype_t))) );
	unsigned medoidDegree = *puNumNeighbours;
	puNeighbour = puNumNeighbours + 1;
	vector<unsigned> medoidNeighbors;

	for(unsigned long long ii = 0; ii < medoidDegree; ++ii)
	{
		medoidNeighbors.push_back(puNeighbour[ii]);
	}


	// [2] Setting the neighbors of medoid as initial candidate neighbors for the query point
	for(unsigned ii=0; ii < numQueries; ++ii ) {
		neighbors[ii * offset] = medoidID; // conside medoid also as initial neighbour
		unsigned numNeighbors = 1; // medoid is already inserted
		for (unsigned i = 0; i < medoidDegree; ++i) {
			neighbors[ii * offset + i + 1] = medoidNeighbors[i];
			numNeighbors++;
		}

		numNeighbors_query[ii] = numNeighbors;
		// Copy the medoid's FP vectorsi Async (row 0) iter = 1 here.
		memcpy(FPSetCoordsList + ((iter-1) * FPSetCoords_rowsize) + (ii*FPSetCoords_size),
				pIndex + (ullIndex_Entry_LEN * medoidID) ,
				FPSetCoords_size_bytes);
	}
	// 0th row in FPSetCoordsListget copied to GPU in Async fashio
	cudaMemcpyAsync(d_FPSetCoordsList + ((iter-1) * FPSetCoords_rowsize),
					FPSetCoordsList + ((iter-1)* FPSetCoords_rowsize),
					FPSetCoords_rowsize_bytes,cudaMemcpyHostToDevice,streamFPTransfers);

	//Transfer neighbor IDs and count to GPU
	gpuErrchk(cudaMemcpy(d_neighbors_temp, neighbors, sizeof(unsigned) * numQueries*(R+1), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_numNeighbors_query_temp, numNeighbors_query, sizeof(unsigned) * (numQueries), cudaMemcpyHostToDevice));

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

	auto milliStart = log_message("SEARCH STARTED");
	auto start = std::chrono::high_resolution_clock::now();
	gpuErrchk(cudaMemcpy(d_queriesFP, queriesFP, sizeof(datatype_t) * (D*numQueries), cudaMemcpyHostToDevice));
	auto stop = std::chrono::high_resolution_clock::now();
	time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;
	gputimer.Start();
	/**
	 * [3] Launching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K1" block size.
	 * One thread block is assigned to a query, i.e., "numThreads_K1" threads perform the computation for a query.
	 */
	populate_pqDist_par<<<numQueries, numThreads_K1, 0, streamKernels>>> (d_pqTable, d_pqDistTables, d_queriesFP, d_chunksOffset, d_centroid, n_chunks);
	gputimer.Stop();
	time_K1 += gputimer.Elapsed();


	// for texture memory
	// for d_compressedVectors
	gputimer.Start();
	gpuErrchk(cudaMemset(d_numNeighbors_query, 0, sizeof(unsigned)*numQueries));
	/** [4] Launching the kernel with "numQueries" number of thread-blocks and block size of 256
	 * One thread block is assigned to a query, i.e., 256 threads perform the computation for a query. The block size has been tuned for performance.
	 */
		#ifdef _DBG_BLOOMFILTER
		if (nQueryID == 0)
		{
			printf("Num Neighbours before filtering : %d \n", numNeighbors_query[nQueryID]);
			for (int i = 0; i< numNeighbors_query[nQueryID]; i++)
				printf ("%u, ", neighbors[i] );

			printf("\n");
		}
		#endif

	neighbor_filtering_new<<<numQueries, numThreads_K5, 0, streamKernels >>> (d_neighbors, d_neighbors_temp, d_numNeighbors_query, d_numNeighbors_query_temp, d_processed_bit_vec);
	gputimer.Stop();
	time_neighbor_filtering += gputimer.Elapsed() ;

	#ifdef _DBG_BLOOMFILTER
	std::set<unsigned> myVisitedSet; // glgobal set to track all visited nodes
	float myUnvisisted_bf = 0.0 ; // what we found as unvisited
	float myUnvisisted_actual = 0.0; // actual truth of unvisited
	unsigned uTempNum = 0;
	unsigned uTempNeighbours[R+1];

	if (nQueryID == 0)
	{
		gpuErrchk(cudaMemcpy(&uTempNum, d_numNeighbors_query, sizeof(unsigned), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(uTempNeighbours, d_neighbors, uTempNum * sizeof(unsigned), cudaMemcpyDeviceToHost));

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

	/** [5] Launching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K2" block size.
	 * One thread block is assigned to a query, i.e., "numThreads_K2" threads perform the computation for a query.
	 */

	compute_neighborDist_par <<<numQueries, numThreads_K2,0, streamKernels >>> (d_neighbors, d_numNeighbors_query, d_compressedVectors,
	d_pqDistTables, d_neighborsDist_query);
	gpuErrchk(cudaMemcpy(d_iter, &iter, sizeof(unsigned), cudaMemcpyHostToDevice));

	/** [6] Launching the kernel with "numQueries" number of thread-blocks and block size of one.
	 * A songle threads perform the computation for a query.
	 * Note: The  additional arithmetic in the number of thread blocks it to arrive a ceil value. After assigning a required value
	 * to numThreads_K4, the division could leat to a truncated quotient, resulting in lesser threads getting spawned.
	 */
	// technically this is start iteration 1, before this it was iteration 0. finding the parent, is start of iteration
	// Neighbor seek if a given parent, processing neighbors, finding next parent is one iteration.
	// iteration 0 occurs outsied the do while loop
	compute_parent1<<<(numQueries + numThreads_K4 -1 )/numThreads_K4,numThreads_K4,0, streamKernels >>>(d_neighbors, d_numNeighbors_query, d_neighborsDist_query,
			d_BestLSets, d_BestLSetsDist, d_BestLSets_visited, d_parents, d_nextIter, d_BestLSets_count, d_mark,
			d_iter,d_L2ParentIds,d_FPSetCoordsList_Counts, d_numQueries);
	gputimer.Stop();
	time_B1_vec.push_back(gputimer.Elapsed());

	// Loop until all the query have no new parent
	do
	{
		// Let's wait for all kernels got a chance to execute before we initiate  transfer of the 'd_parents'
		cudaStreamSynchronize(streamKernels);

		start = std::chrono::high_resolution_clock::now();

		// Transfer parent IDs from GPU to CPU
		gpuErrchk(cudaMemcpyAsync(parents, d_parents, sizeof(unsigned) * ((SIZEPARENTLIST)*numQueries),
									cudaMemcpyDeviceToHost,
									streamParent));


		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;;
		gputimer.Start();
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
										d_mark);
		gputimer.Stop();

		start = std::chrono::high_resolution_clock::now();
		// THis memset is very much required. Though it gets updated but not for queries which don't yield parents
		memset(numNeighbors_query, 0, sizeof(unsigned) * numQueries	);

		cudaStreamSynchronize(streamParent);

		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;


		start = std::chrono::high_resolution_clock::now();
		#pragma omp parallel
		{
			int CPUthreadno =  omp_get_thread_num();
			/*Collecting the node ids of neighborhood of the parent nodes to send to GPU */
			// Note: Only one parent is supported per query. Becasue offset_neighbors is function of
			// queryID only. To support multiple parents in a query, it should have been dependent on
			// parent number(jj) within a query
			for(unsigned ii=CPUthreadno; ii < numQueries; ii = ii + numCPUthreads)
			{
				unsigned numParents = parents[ii*(SIZEPARENTLIST)];
				#ifdef _DBG_BOUNDS

				if (numParents > 1)
					printf("ERROR : numParents of %u iter=%u\n", numParents, iter);
				#endif

				for(unsigned jj = 1; jj <= numParents; ++jj)
				{
					unsigned curreParent = parents[ii*(SIZEPARENTLIST)+jj];;
					#ifdef _DBG_VERBOSE
					if (ii == nQueryID)
					{
						printf("\n ==> [%d] CPU side: Parent = %d\n", iter, curreParent);
						//fileParents.write((char*)&(curreParent), sizeof(unsigned) );
					}
					#endif
					unsigned offset_neighbors = ii * offset;

					// Copy the Parent's'FP vectors of current query
					memcpy(FPSetCoordsList + (iter * FPSetCoords_rowsize) + (ii*FPSetCoords_size),
							pIndex + (ullIndex_Entry_LEN * curreParent),
							FPSetCoords_size_bytes);

					// Extract Neighbour
					unsigned *puNumNeighbours = (unsigned*)(pIndex + ((ullIndex_Entry_LEN*curreParent)+(sizeof(datatype_t)*D)) );;

					#ifdef _DBG_BOUNDS
					// validation
					if (*puNumNeighbours > 64)
						printf("*** ERROR: NumNeighbours %d which is > 64\n",*puNumNeighbours);
					#endif

					memcpy(neighbors+offset_neighbors, puNumNeighbours+1, (*puNumNeighbours) * sizeof(unsigned));
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

		gpuErrchk(cudaMemcpyAsync(d_neighbors_temp, neighbors, sizeof(unsigned) * numQueries*(R+1),
									cudaMemcpyHostToDevice,
									streamChildren));

		gpuErrchk(cudaMemcpyAsync(d_numNeighbors_query_temp, numNeighbors_query, sizeof(unsigned) * (numQueries),
									cudaMemcpyHostToDevice,
									streamChildren));

		// Transfer the FP vectors also from CPU to GPU in Async fashion
		cudaMemcpyAsync(d_FPSetCoordsList + (iter * FPSetCoords_rowsize),
					FPSetCoordsList + (iter * FPSetCoords_rowsize),
					FPSetCoords_rowsize_bytes, cudaMemcpyHostToDevice, streamFPTransfers);

		gpuErrchk(cudaMemsetAsync(d_numNeighbors_query, 0, sizeof(unsigned)*numQueries, streamChildren));
		cudaStreamSynchronize(streamChildren);
		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

		#ifdef _DBG_BLOOMFILTER
		std::set<unsigned> myNeigbours; //local set to track Neigbours (overwritten on every iteration)

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
		neighbor_filtering_new<<<numQueries, numThreads_K5,0, streamKernels >>> (d_neighbors,
																		d_neighbors_temp,
																		d_numNeighbors_query,
																		d_numNeighbors_query_temp,
																		d_processed_bit_vec);
		gputimer.Stop();
		time_neighbor_filtering += gputimer.Elapsed() ;

		#ifdef _DBG_BLOOMFILTER
		unsigned uTempNum = 0;
		unsigned uTempNeighbours[R+1];

		if (nQueryID == 0)
		{
			// Lets filter using std::set and compare with our Bloom Filter
			std::set<int> myResult ;
			set_difference(myNeigbours.begin(),  myNeigbours.end(),
							myVisitedSet.begin(), myVisitedSet.end(),
							std::inserter(myResult, myResult.end()) );

			myVisitedSet.insert(myResult.begin(), myResult.end());

			gpuErrchk(cudaMemcpy(&uTempNum, d_numNeighbors_query, sizeof(unsigned), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(uTempNeighbours, d_neighbors, uTempNum * sizeof(unsigned), cudaMemcpyDeviceToHost));

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
		gpuErrchk(cudaMemsetAsync(d_neighborsDist_query,0,sizeof(float) * (numQueries*(R+1)), streamKernels ));

		/** [12]vLaunching the kernel with "numQueries" number of thread-blocks and user specified "numThreads_K2" block size.
		 * One thread block is assigned to a query, i.e., "numThreads_K2" threads perform the computation for a query.
		 */

		compute_neighborDist_par <<<numQueries, numThreads_K2,0,streamKernels >>> (d_neighbors, d_numNeighbors_query, d_compressedVectors,
		d_pqDistTables, d_neighborsDist_query);

		++iter;

		gpuErrchk(cudaMemcpy(d_iter, &iter, sizeof(unsigned), cudaMemcpyHostToDevice));

		/** [13] Launching the kernel with "numQueries" number of thread-blocks and block size of one.
	 	* A songle threads perform the computation for a query.
		 */
		// previous d_iter would have transferred by now
		compute_parent2<<<(numQueries + numThreads_K4 -1 )/numThreads_K4,numThreads_K4,0, streamKernels >>>(d_neighbors, d_numNeighbors_query, d_neighborsDist_query,  d_BestLSets,
				d_BestLSetsDist, d_BestLSets_visited, d_parents, d_nextIter, d_BestLSets_count, d_mark,
				d_iter, d_L2ParentIds, d_FPSetCoordsList_Counts, d_numQueries);

		gputimer.Stop();
		time_B1_vec.push_back(gputimer.Elapsed());
		start = std::chrono::high_resolution_clock::now();

		gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));  //d_nextIter calculated in compute_parent<<< >>>
		// Note: Default Stream operations (cmputation or memory transfers) cannot overlap with operatiosn on other sterrams.
		// Hence, the above call could act as a synchronization mechanism to ensure all kernels are done (next parent ready)
		// before we start seeking neighbours on CPU

		stop = std::chrono::high_resolution_clock::now();
		time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

		#ifdef _DBG_BOUNDS
		if (iter == MAX_PARENTS_PERQUERY-1)
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
	gpuErrchk(cudaFree(d_compressedVectors));
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
	//gpuErrchk(cudaFree(d_merged_visited));
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

	// needed till iterations end
	cudaFreeHost(neighbors);
	neighbors = NULL;
	cudaFreeHost(numNeighbors_query);
	numNeighbors_query = NULL;
	cudaFreeHost(parents);
	parents = NULL;
#endif // #if FREE_AFTERUSE


	// re-rnking start
	gputimer.Start();

	cudaStreamSynchronize(streamFPTransfers);
	compute_L2Dist<<<numQueries, K4_blockSize >>> (d_FPSetCoordsList,
												d_FPSetCoordsList_Counts,
												d_queriesFP,
												d_L2ParentIds,
												d_L2distances,
												d_nearestNeighbours,
												d_numQueries);

#ifdef _DBG_RERANKING
{
	cudaDeviceSynchronize();
	float* pTempDists = (float*)malloc(sizeof(float) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempDists, d_L2distances, sizeof(float) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	unsigned* pTempParents = (unsigned*)malloc(sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempParents, d_L2ParentIds, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	unsigned* pTempNumParents = (unsigned*)malloc(sizeof(unsigned) * numQueries);
	gpuErrchk(cudaMemcpy(pTempNumParents, d_FPSetCoordsList_Counts, sizeof(unsigned) * numQueries,
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
	compute_NearestNeighbours<<<numQueries, MAX_PARENTS_PERQUERY >>> (d_L2ParentIds,
												d_L2ParentIds_aux,
												d_FPSetCoordsList_Counts,
												d_L2distances,
												d_L2distances_aux,
												d_nearestNeighbours,
												d_numQueries,
												d_recall);



#ifdef _DBG_RERANKING
{
	cudaDeviceSynchronize();
	float* pTempDists = (float*)malloc(sizeof(float) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempDists, d_L2distances, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	unsigned* pTempParents = (unsigned*)malloc(sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries));
	gpuErrchk(cudaMemcpy(pTempParents, d_L2ParentIds, sizeof(unsigned) * (MAX_PARENTS_PERQUERY * numQueries),
	cudaMemcpyDeviceToHost));

	unsigned* pTempNumParents = (unsigned*)malloc(sizeof(unsigned) * numQueries);
	gpuErrchk(cudaMemcpy(pTempNumParents, d_FPSetCoordsList_Counts, sizeof(unsigned) * numQueries,
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

		unsigned* pTempLSet = (unsigned*)malloc(sizeof(unsigned) * (L * numQueries));
		gpuErrchk(cudaMemcpy(pTempLSet, d_BestLSets, sizeof(unsigned) * (L * numQueries),
		cudaMemcpyDeviceToHost));

		float* pTempLSetDist = (float*)malloc(sizeof(float) * (L * numQueries));
		gpuErrchk(cudaMemcpy(pTempLSetDist, d_BestLSetsDist, sizeof(float) * (L * numQueries),
		cudaMemcpyDeviceToHost));

		unsigned* pTempLSetCount = (unsigned*)malloc(sizeof(unsigned) * (  numQueries));
		gpuErrchk(cudaMemcpy(pTempLSetCount, d_BestLSets_count, sizeof(unsigned) * ( numQueries),
		cudaMemcpyDeviceToHost));
		printf("Best_L_Set : \n");
		for (int i =0; i < pTempLSetCount[nQueryID]; i++)
				printf("Parent = %d \t distance = %f\n", pTempLSet[(nQueryID*L) + i], pTempLSetDist[(nQueryID*L) + i] );
		free(pTempLSet);
		free(pTempLSetDist);
		free(pTempLSetCount);


#endif

	start = std::chrono::high_resolution_clock::now();

	gpuErrchk(cudaMemcpy(nearestNeighbours, d_nearestNeighbours, sizeof(unsigned) * (recall_at * numQueries),
				cudaMemcpyDeviceToHost));
	stop = std::chrono::high_resolution_clock::now();
	time_transfer += std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() / 1000.0;

	// re-rnking end


	// Copy the BestLSets to CPU
	unsigned* BestLSets = (unsigned*)malloc(sizeof(unsigned) * (L * numQueries));
	gpuErrchk(cudaMemcpy(BestLSets, d_BestLSets, sizeof(unsigned) * (L * numQueries),
	cudaMemcpyDeviceToHost));

	auto milliEnd = log_message("SEARCH END");
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
				// use best L-Set instead directly for ANNs instead of re-ranking
			}

			#pragma omp critical
			{
				final_bestL1[ii] = query_best;
			}
		}
	}



#ifdef FREE_AFTERUSE
	free(queriesFP);
	queriesFP = NULL;
	gpuErrchk(cudaFreeHost(FPSetCoordsList));
	free(nearestNeighbours);
	nearestNeighbours = NULL;

	gpuErrchk(cudaFree(d_queriesFP));
	gpuErrchk(cudaFree(d_FPSetCoordsList));
	gpuErrchk(cudaFree(d_FPSetCoordsList_Counts));
	gpuErrchk(cudaFree(d_L2distances));
	gpuErrchk(cudaFree(d_L2ParentIds));
	gpuErrchk(cudaFree(d_nearestNeighbours));
	gpuErrchk(cudaFree(d_recall));
	gpuErrchk(cudaFree(d_numQueries));

	cudaStreamDestroy(streamFPTransfers);
	cudaStreamDestroy(streamKernels);
#endif // #if FREE_AFTERUSE

	cout << "iterations = " <<  iter << endl;

	assert(time_B1_vec.size() >= 1);
	float time_B1_avg = time_B1_vec[0];
	time_B1 = time_B1_avg;
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
#ifdef _DBG_BOUNDS
	unsigned* FPSetCoordsList_Counts_temp = (unsigned*)malloc(sizeof(unsigned) * numQueries);

	gpuErrchk(cudaMemcpy(FPSetCoordsList_Counts_temp, d_FPSetCoordsList_Counts, sizeof(unsigned) * numQueries, cudaMemcpyDeviceToHost ));
	for (int i=0; i< numQueries; i++)
	{
		total_size += FPSetCoordsList_Counts_temp[i];
	}
	free(FPSetCoordsList_Counts_temp);
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
		cout << L1 << "\t" << recall << "\t total parents " << total_size << endl;
	}

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

	char c = 'n';
	cout << "Try Next run ? [y|n]" << endl;
	cin >> c;
	if (c !='y')
		break;

}// outer do-while for running the search multiple times. The stats across runs can be averaged etc
// for a reporting results in a consistent manner (i.e. avoid affect of outliers)
while (1);
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
__global__ void populate_pqDist_par(float* d_pqTable_T, float* d_pqDistTables, datatype_t* d_queriesFP, unsigned* d_chunksOffset, float* d_centroid, unsigned n_chunks)
 {
 	// [kv] what happens if shared memory overflows in SIFT 1BN
	__shared__ datatype_t query_vec[D];
	__shared__ float shm_centroid[D];
	unsigned queryID = blockIdx.x;
	unsigned pqDistTables_offset = queryID * 256 * n_chunks;	// Offset to the beginning of the pqTable entries for this query

	unsigned gid = queryID * D;
	unsigned tid = threadIdx.x;

	for(unsigned i= tid; i < D; i += blockDim.x) {
		query_vec[i] = d_queriesFP[gid + i];
		shm_centroid[i] = d_centroid[i];
	}

	__syncthreads();

	// Calculate and place 256 entries in the pqDistTables array corresponding to the each chunk
	for (unsigned chunk = 0; chunk < n_chunks; chunk++) {
		unsigned chunk_start = pqDistTables_offset + (256 * chunk);	// Size of pqDistTables = (numQueries * n_chunks * 256)
		for (unsigned j = d_chunksOffset[chunk]; j < d_chunksOffset[chunk + 1]; j++) { 	// if a chunk contains 4-dimensions
			const float *centers_dim_vec = d_pqTable_T + (256 * j);						// then j=0,1,2,3   4,5,6,7
			// [kv] should 256 be hard-coded or based on the dim
			for (unsigned idx = tid; idx < 256; idx += blockDim.x) {					//// Memory Coalescing
				float diff = centers_dim_vec[idx] - ((datatype_t) query_vec[j] - shm_centroid[j]);

				d_pqDistTables[chunk_start + idx] += (float)(diff * diff);
			}
		}
	}
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
										bool* d_processed_bit_vec) {
	unsigned queryID = blockIdx.x;
	unsigned tid = threadIdx.x;

	unsigned offset_neighbors = queryID * (R+1); //Offset into d_neighbors_temp array
	unsigned offset_bit_vec = queryID*BF_MEMORY;	//Offset into d_processed_bit_vec vector of bloom filter
	unsigned numNeighbors = d_numNeighbors_query_temp[queryID];
	bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;

	// For each neighbor in d_neighbors_temp array check if its corresponding bits in the d_processed_bit_vec are already set
	for(unsigned ii=tid; ii < numNeighbors; ii += blockDim.x ) {
		unsigned nbr = d_neighbors_temp[offset_neighbors+ii];
		if(!((d_processed_bit_vec_start[hashFn1_d(nbr)]) && (d_processed_bit_vec_start[ hashFn2_d(nbr)])))
		{
			d_processed_bit_vec_start[ hashFn1_d(nbr)] = true;	//Set the bit to true
			d_processed_bit_vec_start[ hashFn2_d(nbr)] = true;	//Set the bit to true
			unsigned old = atomicAdd(&d_numNeighbors_query[queryID], 1);
			d_neighbors[offset_neighbors + old] = nbr;
		}
	}
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


/**
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
// The threadblock size is 8*R = 8*64 = 512. Each thread will compute dist on CHUNKS/8 dimensions
	#define THREADS_PER_NEIGHBOR 8

__global__ void  compute_neighborDist_par(unsigned* d_neighbors,
											unsigned* d_numNeighbors_query,
											uint8_t* d_compressedVectors,
											float* d_pqDistTables,
											float*  d_neighborsDist_query) {
	unsigned tid = threadIdx.x;
	unsigned queryID = blockIdx.x;
	unsigned numNeighbors = d_numNeighbors_query[queryID];
	unsigned pqDistTables_offset = queryID * 256 * CHUNKS;
	float* d_pqDistTables_start = d_pqDistTables + pqDistTables_offset;

	unsigned queryNeighbors_offset  = queryID * (R+1);	// offset into d_neighbors array
	unsigned* d_neighbors_start  = d_neighbors + queryNeighbors_offset;
	float* d_neighborsDist_query_start = d_neighborsDist_query + queryNeighbors_offset;
	uint8_t* d_compressedVectors_start = NULL;
	float sum = 0.0f;
	for(unsigned uIter = tid; uIter < numNeighbors; uIter += blockDim.x)
		d_neighborsDist_query_start[uIter] = 0;

	typedef cub::WarpReduce<float,THREADS_PER_NEIGHBOR> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage[R];

	for( unsigned j = tid/THREADS_PER_NEIGHBOR; j < numNeighbors; j += (blockDim.x)/THREADS_PER_NEIGHBOR ) { // assign eight threads to a neighbor, within a query


		d_compressedVectors_start = d_compressedVectors + ((unsigned long long)d_neighbors_start[j]) *CHUNKS;
		sum = 0.0f;

		for(unsigned long long i = tid%THREADS_PER_NEIGHBOR; i < CHUNKS; i += THREADS_PER_NEIGHBOR ){
			sum += d_pqDistTables_start[(i * 256) + d_compressedVectors_start[i]];
		}

		d_neighborsDist_query_start[j] = WarpReduce(temp_storage[j]).Sum(sum);
	}
}
__global__ void compute_L2Dist (datatype_t* d_FPSetCoordsList,
								unsigned* d_FPSetCoordsList_Counts,
								datatype_t* d_queriesFP,
								unsigned* d_L2ParentIds,
								float* d_L2distances,
								unsigned* d_nearestNeighbours,
								unsigned* d_numQueries)
{

	__shared__ datatype_t query_vec[D];
	//datatype_t query_vec[D];
	unsigned queryID = blockIdx.x;
	unsigned numNodes = d_FPSetCoordsList_Counts[queryID];
	unsigned tid = threadIdx.x;
	unsigned gid = queryID * D;
	unsigned numQueries = *d_numQueries;
	const unsigned long long FPSetCoords_size = D; // * sizeof(datatype_t) Note : To be used for array indexing, hence not byte offsets
	const unsigned long long FPSetCoords_rowsize = FPSetCoords_size * numQueries;

	for(unsigned ii= tid; ii < D; ii += blockDim.x) {
		query_vec[ii] = d_queriesFP[gid + ii];
	}
	__syncthreads();

	// one thread block computes the distances of all the nodes for a query,
	for(unsigned ii = tid; ii < numNodes; ii += blockDim.x) {
		float L2Dist = 0.0;
		for(unsigned jj=0; jj < D; ++jj) {
			float diff = d_FPSetCoordsList[(FPSetCoords_rowsize * ii) + (queryID * FPSetCoords_size) + jj] - query_vec[jj];
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
						unsigned* d_recall)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
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
	for(unsigned ii = tid; ii < *d_recall; ii += blockDim.x)
	{
		d_nearestNeighbours[( (*d_numQueries) * ii) + queryID ] = d_L2ParentIds[( (*d_numQueries) * ii) + queryID ];
	}
}

 __global__ void  compute_parent1(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
 								unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
 								unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
 								unsigned* d_mark,
 								unsigned* d_iter,
 								unsigned* d_L2ParentIds,
 								unsigned* d_FPSetCoordsList_Counts,
 								unsigned* d_numQueries)
 {
	unsigned tid = threadIdx.x;
  	unsigned queryID = (blockIdx.x*blockDim.x) + tid;
	if (queryID >= (*d_numQueries) )
		return;

  	// Single thread is enough to compute the parent (pre-fetched) for a query. So, One ThreadBlock can work
  	// on computing parents for blockDim.x Queries
	unsigned numNeighbors = d_numNeighbors_query[queryID];
	//unsigned Best_L_Set_size = d_BestLSets_count[queryID];

	float dist=3.402823E+38;	//assigning max float value
	unsigned index = 0;
	unsigned numQueries = *d_numQueries;

	unsigned offset = queryID *(R+1);

	/*Locate closest neighbor in neighbor list*/
	//  Note: MEDOID is excluded becasue we already fetched its children in the very beginning
	for(unsigned ii=0; ii < numNeighbors; ++ii) {
		if(d_neighborsDist_query[offset + ii] < dist && d_neighbors[offset+ii]!= MEDOID){
			index=offset+ii;
			dist=d_neighborsDist_query[index];
		}
	}

	unsigned parentIndex = 0;

		parentIndex++;
		d_parents[queryID*(SIZEPARENTLIST)+parentIndex] = d_neighbors[index];
		d_mark[queryID]= d_neighbors[index];
		// Lets mark this nodeID for re-ranking later
		//d_basepoints_parentqueries[d_neighbors[index]] = queryID;


	// Place the parent in d_parents array if parent is decided and set the next iteration flag to true
	// indicates the numParents used in neighbours seeking step
	d_parents[queryID*(SIZEPARENTLIST)] = parentIndex;
	// Note: parentIndex should ideally be accessed in a synchronized manner.
	// (but only WRITEs (no READs) are there now,   so its ok)

	*d_nextIter = true;
	// Note: One thread assigned to one Query, so ok to increment (no contention)
	d_FPSetCoordsList_Counts[queryID]++;
	d_L2ParentIds[( (*d_iter) * numQueries) + queryID] = d_parents[queryID*(SIZEPARENTLIST)+parentIndex];


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

 __global__ void  compute_parent2(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
 								unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
 								unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
 								unsigned* d_mark,
 								unsigned* d_iter,
 								unsigned* d_L2ParentIds,
 								unsigned* d_FPSetCoordsList_Counts,
 								unsigned* d_numQueries)
 {
	unsigned tid = threadIdx.x;
  	unsigned queryID = (blockIdx.x*blockDim.x) + tid;
	if (queryID >= (*d_numQueries) )
		return;

  	// Single thread is enough to compute the parent (pre-fetched) for a query. So, One ThreadBlock can work
  	// on computing parents for blockDim.x Queries
	unsigned numNeighbors = d_numNeighbors_query[queryID];
	unsigned Best_L_Set_size = d_BestLSets_count[queryID];

	float dist=3.402823E+38;	//assigning max float value
	unsigned index = 0;
	unsigned numQueries = *d_numQueries;
	unsigned offset = queryID*(R+1);

	/*Locate closest neighbor in neighbor list*/
	//  Note: MEDOID is excluded becasue we already fetched its children in the very beginning
	for(unsigned ii=0; ii < numNeighbors; ++ii) {
		if(d_neighborsDist_query[offset + ii] < dist && d_neighbors[offset+ii]!= MEDOID){
			index=offset+ii;
			dist=d_neighborsDist_query[index];
		}
	}
	unsigned parentOffset = queryID*(SIZEPARENTLIST);
	unsigned parentIndex = 0;
	unsigned LOffset = L*queryID;
	unsigned LIndex = LOffset;

	/*Compare closest neighbor in neighbor list with first unvisited node in Best_L_Set*/
	for(unsigned ii=0; ii < Best_L_Set_size; ++ii) {
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
		d_L2ParentIds[( (*d_iter) * numQueries) + queryID] = d_parents[parentOffset + parentIndex];
	}
	#ifdef _DBG_ITERATIONS
	else
	{
		if (tid==0)
		{
			// print the
			if (*d_iter == d_FPSetCoordsList_Counts[queryID])
				printf("Query ID = %u and #iteration = %d\n", queryID,d_FPSetCoordsList_Counts[queryID]);
				// No of parents indicates  the number loops/work for that query. Hence printing it as a measure of how long a query runs
		}
	}
	#endif
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
__global__ void  compute_BestLSets_par_sort_msort(unsigned* d_neighbors,
													unsigned* d_neighbors_aux,
													unsigned* d_numNeighbors_query,
													float* d_neighborsDist_query,
													float* d_neighborsDist_query_aux,
													bool* d_nextIter) {


	unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    *d_nextIter = false;

    if(tid >= numNeighbors || numNeighbors <= 0) return;

    __shared__ unsigned shm_pos[R+1];
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
											unsigned* d_mark){
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


                nbrsBound = max(nbrsBound, min(L-Best_L_Set_size, numNeighbors));
                // if both Best_L_Set_size and numNeighbors is less than L, then the max of the two will be the newBest_L_Set_size otherwise it will be L
                newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, L);

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

		// Mark the node extracted by compute_parent kernel as visited
        for(int i=tid;i<newBest_L_Set_size;i=i+blockDim.x){
                if(d_mark[queryID]==d_BestLSets[L*queryID+tid])
                        d_BestLSets_visited[L*queryID+tid]=true;
        }
}



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
