#ifndef PARANN_H_
#define PARANN_H_

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <sys/stat.h>
#include <vector>
#include <map>


#define ROUND_UP(X, Y) \
	((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

using std::string;

/*
For Plotting results for dataset

1) Fix the L and BF size for the "right-most point" (point with lowest QPS and highest recall).
BF size is calculated based on how much memory is remaining on GPU

2) Then start varying the L to generates points towards left and north on the plot

3) define the dataset-specific #define like done below for each dataset below.
ToDo: L will be made as an interactive parameter to avoid re-compilations

4) Discard the first run results as it will be higher (outlier). Take the next 5 runs and
compute the geomean.

5) Turn OFF stats break-up using ENABLE_GPU_STATS in commandline

*/


#define SIFT100M
#define L 40 // L_search
#define CHUNKS 64




#ifdef DEEP1BSMALL
typedef float datatype_t; // go to header file to change the datatype
#define INDEX_ENTRY_LEN (644)

#define D 96 // dimensions of the data
#define L 10 // L_search
#define CHUNKS 96
#define MEDOID 978 // sift 1B
#define N 10000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 128!
#endif

#ifdef DEEP1B
typedef float datatype_t; // go to header file to change the datatype
#define INDEX_ENTRY_LEN (644)

#define D 96 // dimensions of the data
#define L 200     // L_search
#define CHUNKS 74 // # of chunks for SIFT1b
#define MEDOID 178757270 // sift 1B
//#define MEDOID 642390445 //0 // sift 1B BFS
#define N 1000000000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1
#endif

#ifdef SIFT1BTOY
typedef uint8_t datatype_t; // go to header file to change the datatype
#define INDEX_ENTRY_LEN (14)
#define D 2 // dimensions of the data
#define L 4 // L_search
#define CHUNKS 2
#define MEDOID 6 // sift 1B
#define N 14// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 128!
#endif

#ifdef SIFT1BSMALL
typedef uint8_t datatype_t; // go to header file to change the datatype
#define INDEX_ENTRY_LEN (388)

#define D 128 // dimensions of the data
#define L 10 // L_search
#define CHUNKS 128
#define MEDOID 7999 // sift 1B
#define N 10000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 128!
#endif


#ifdef SIFT1B
typedef uint8_t datatype_t; // go to header file to change the datatype
#define INDEX_ENTRY_LEN (388)

#define D 128 	// dimensions of the data
#define L 152   // L_search (FIXED)

#define CHUNKS 74 // # of chunks for SIFT1b
#define MEDOID 178757270 // sift 1B
//#define MEDOID 642390445 //0 // sift 1B BFS
#define N 1000000000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1
#endif


#ifdef SIFT1M

typedef float datatype_t;
#define INDEX_ENTRY_LEN (772)

#define D 128 // dimensions of the data

#define MEDOID 123742// 221898 // sift 1B
#define N 1000000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


#ifdef GIST1M

typedef float datatype_t;
#define INDEX_ENTRY_LEN (4100)

#define D 960 // dimensions of the data

#define MEDOID 356422// 221898 // sift 1B
#define N 1000000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


#ifdef GLOVE200

typedef float datatype_t;
#define INDEX_ENTRY_LEN (1060)

#define D 200 // dimensions of the data

#define MEDOID 716135// 221898 // sift 1B
#define N 1183514// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


#ifdef NYTIMES

typedef float datatype_t;
#define INDEX_ENTRY_LEN (1284)

#define D 256 // dimensions of the data

#define MEDOID 221898 // 221898 // sift 1B
#define N 289761// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


#ifdef SIFT100M

typedef uint8_t datatype_t;
#define INDEX_ENTRY_LEN (388)

#define D 128 // dimensions of the data

#define MEDOID 59689614 // 65610822 // 221898 // sift 1B
#define N 100000000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


#ifdef MNIST8M

typedef uint8_t datatype_t;
#define INDEX_ENTRY_LEN (1044)

#define D 784 // dimensions of the data

#define MEDOID 2096858 // 221898 // sift 1B
#define N 8090000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


#ifdef DEEP100M

typedef float datatype_t;
#define INDEX_ENTRY_LEN (644)

#define D 96 // dimensions of the data

#define MEDOID 68983084 // 221898 // sift 1B
#define N 100000000// total number of nodes in the dataset
#define NUMTHREADS_COMPUTEPARENT 1  // surprisingly, found 1 to work better than 100!
#endif


__global__ void populate_pqDist_par(float *d_pqTable, float* d_pqDistTables, datatype_t* d_queriesFP, unsigned* d_chunksOffset, float* d_centroid, unsigned n_chunks);


__global__ void  compute_neighborDist_par(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float*  d_neighborsDist_query, datatype_t* d_queriesFP, uint8_t* d_pIndex);

__global__ void  compute_parent1(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
							unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
							unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
							unsigned* d_mark,
							unsigned* d_iter,
 							unsigned* d_L2ParentIds,
 							unsigned* d_FPSetCoordsList_Counts,
 							unsigned* d_numQueries);

__global__ void  compute_parent2(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query,
							unsigned* d_BestLSets, float* d_BestLSetsDist, bool* d_BestLSets_visited,
							unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
							unsigned* d_mark,
							unsigned* d_iter,
 							unsigned* d_L2ParentIds,
 							unsigned* d_FPSetCoordsList_Counts,
 							unsigned* d_numQueries);

__global__ void  compute_BestLSets_par_sort_msort(unsigned* d_neighbors, unsigned* d_neighbors_aux, unsigned* d_neighbors_offset, float* d_neighborsDist_query, float* d_neighborsDist_query_aux,  bool* d_nextIter);

__global__ void  compute_BestLSets_par_merge(unsigned* d_neighbors, unsigned* d_numNeighbors_query, float* d_neighborsDist_query, unsigned* d_BestLSets, float* d_BestLSetsDist,	bool* d_BestLSets_visited, unsigned* d_parents, unsigned iter, bool* d_nextIter, unsigned* d_BestLSets_count, unsigned* d_L2ParentIds, unsigned* d_FPSetCoordsList_Counts, unsigned* d_numQueries);

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
																								
													);

__global__ void neighbor_filtering_new (unsigned* d_neighbors,
										unsigned* d_neighbors_temp,
										unsigned* d_numNeighbors_query,
										unsigned* d_numNeighbors_query_temp,
										bool* d_processed_bit_vec,
										unsigned* d_parents,
										uint8_t* d_pIndex,
										unsigned iter,
										bool* d_nextIter);


__global__ void compute_L2Dist (datatype_t* d_FPSetCoordsList,
								unsigned* d_FPSetCoordsList_Counts,
								datatype_t* d_queriesFP,
								unsigned* d_L2ParentIds,
								float* d_L2distances,
								unsigned* d_nearestNeighbours,
								unsigned* d_numQueries,
								uint8_t* d_pIndex);

__global__ void  compute_NearestNeighbours(unsigned* d_L2ParentIds,
						unsigned* d_L2ParentIds_aux,
						unsigned* d_FPSetCoordsList_Counts,
						float* d_L2distances,
						float* d_L2distances_aux,
						unsigned* d_nearestNeighbours,
						unsigned* d_numQueries,
						unsigned* d_recall,
						unsigned* d_BestLSets);

__global__ void  compute_neighborDist_par_cachewarmup(unsigned* d_neighbors,
											uint8_t* d_compressedVectors);

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
  //int nLevel;
} Node; // size = 8 bytes

typedef std::vector<unsigned> NeighbourList;
typedef std::map<unsigned,Node*> NodeIDMap;
void SetupBFS(NodeIDMap& p_mapNodeIDToNode);
void ExitBFS(NodeIDMap& p_mapNodeIDToNode);
void bfs(unsigned uMedoid,
		const unsigned nNodesToDiscover,
		unsigned& visit_counter,
		NodeIDMap& mapNodeIDToNode,
		uint8_t* pGraph);

class cached_ifstream {
	public:
		cached_ifstream() {
		}
		cached_ifstream(const std::string& filename, uint64_t cacheSize)
			: cache_size(cacheSize), cur_off(0) {
				this->open(filename, cache_size);
			}
		~cached_ifstream() {
			delete[] cache_buf;
			reader.close();
		}

		void open(const std::string& filename, uint64_t cacheSize);
		size_t get_file_size();

		void read(char* read_buf, uint64_t n_bytes);
	private:
		// underlying ifstream
		std::ifstream reader;
		// # bytes to cache in one shot read
		uint64_t cache_size = 0;
		// underlying buf for cache
		char* cache_buf = nullptr;
		// offset into cache_buf for cur_pos
		uint64_t cur_off = 0;
		// file size
		uint64_t fsize = 0;
};

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



double calculate_recall(unsigned num_queries, unsigned *gold_std,
		float *gs_dist, unsigned dim_gs,
		unsigned *our_results, unsigned dim_or,
		unsigned recall_at);

/*Inline functions must be defined in the .h file*/

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	auto        val = stat(name.c_str(), &buffer);
	std::cout << " Stat(" << name.c_str() << ") returned: " << val
		<< std::endl;
	return (val == 0);
}


inline void alloc_aligned(void** ptr, size_t size, size_t align) {
	*ptr = nullptr;
	assert(IS_ALIGNED(size, align));
#ifndef _WINDOWS
	*ptr = ::aligned_alloc(align, size);
#else
	*ptr = ::_aligned_malloc(size, align);  // note the swapped arguments!
#endif
	assert(*ptr != nullptr);
}


// compute ground truth

inline void load_truthset(const std::string& bin_file, uint32_t*& ids,
		float*& dists, size_t& npts, size_t& dim) {
	uint64_t           read_blk_size = 64 * 1024 * 1024;
	cached_ifstream reader(bin_file, read_blk_size);
	std::cout << "Reading truthset file " << bin_file.c_str() << " ..."
		<< std::endl;
	size_t actual_file_size = reader.get_file_size();

	int npts_i32, dim_i32;
	reader.read((char*) &npts_i32, sizeof(int));
	reader.read((char*) &dim_i32, sizeof(int));
	npts = (unsigned) npts_i32;
	dim = (unsigned) dim_i32;

	std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
		<< std::endl;

	size_t expected_actual_file_size =
		2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);
	if (actual_file_size != expected_actual_file_size) {
		std::stringstream stream;
		stream << "Error. File size mismatch. Actual size is " << actual_file_size
			<< " while expected size is  " << expected_actual_file_size
			<< " npts = " << npts << " dim = " << dim << std::endl;
		std::cout << stream.str();
		//      throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
		//                                  __LINE__);
		exit(1);
	}

	ids = new uint32_t[npts * dim];
	reader.read((char*) ids, npts * dim * sizeof(uint32_t));
	dists = new float[npts * dim];
	reader.read((char*) dists, npts * dim * sizeof(float));
}

template<typename T>
inline void load_aligned_bin_impl(std::basic_istream<char>& reader,
		size_t actual_file_size, T*& data,
		size_t& npts, size_t& dim,
		size_t& rounded_dim) {
	int npts_i32, dim_i32;
	reader.read((char*) &npts_i32, sizeof(int));
	reader.read((char*) &dim_i32, sizeof(int));
	npts = (unsigned) npts_i32;
	dim = (unsigned) dim_i32;

	size_t expected_actual_file_size =
		npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
	if (actual_file_size != expected_actual_file_size) {
		std::stringstream stream;
		stream << "Error. File size mismatch. Actual size is " << actual_file_size
			<< " while expected size is  " << expected_actual_file_size
			<< " npts = " << npts << " dim = " << dim
			<< " size of <T>= " << sizeof(T) << std::endl;
		std::cout << stream.str() << std::endl;
		//  throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
		//                             __LINE__);
		exit(1);
	}
	rounded_dim = ROUND_UP(dim, 8);
	std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
		<< ", aligned_dim = " << rounded_dim << "..." << std::flush;
	size_t allocSize = npts * rounded_dim * sizeof(T);
	std::cout << "allocating aligned memory, " << allocSize << " bytes..."
		<< std::flush;
	alloc_aligned(((void**) &data), allocSize, 8 * sizeof(T));
	std::cout << "done. Copying data..." << std::flush;

	for (size_t i = 0; i < npts; i++) {
		reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
		memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
	}
	std::cout << " done." << std::endl;
}


template<typename T>
inline void load_aligned_bin(const std::string& bin_file, T*& data,
		size_t& npts, size_t& dim, size_t& rounded_dim) {
	std::cout << "Reading bin file " << bin_file << " ..." << std::flush;
	// START OLS
	//_u64            read_blk_size = 64 * 1024 * 1024;
	// cached_ifstream reader(bin_file, read_blk_size);
	// size_t actual_file_size = reader.get_file_size();
	// END OLS

	std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
	uint64_t      fsize = reader.tellg();
	reader.seekg(0);
	load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim);
}


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
