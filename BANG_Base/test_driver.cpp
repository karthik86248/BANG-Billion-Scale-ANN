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
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
#include <sys/stat.h>
#include <vector>
#include <set>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#include "bang.h"
using namespace std;

double calculate_recall(unsigned num_queries, unsigned *gold_std,
		float *gs_dist, unsigned dim_gs,
		result_ann_t *our_results, unsigned dim_or,
		unsigned recall_at) {
	double             total_recall = 0;
	std::set<unsigned> gt, res;

	for (size_t i = 0; i < num_queries; i++) {
		//cout << "Query : " << i << endl;
		gt.clear();
		res.clear();
		unsigned *gt_vec = gold_std + dim_gs * i;
		result_ann_t *res_vec = our_results + dim_or * i;
		size_t    tie_breaker = recall_at;

		if (gs_dist != nullptr) {
			tie_breaker = recall_at - 1;
			float *gt_dist_vec = gs_dist + dim_gs * i;
			while (tie_breaker < dim_gs &&
					gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
				tie_breaker++;
		}

		gt.insert(gt_vec, gt_vec + tie_breaker);
		res.insert(res_vec, res_vec + recall_at);
		/*cout << "Results: ";
		for (int nIterInner = 0; nIterInner < recall_at; nIterInner++)
		{
			cout << our_results[(i*recall_at) + nIterInner] << "\t" ;
		}
		cout << endl;
		cout << "GT: ";*/
		unsigned cur_recall = 0;
		for (auto &v : gt) {
			//cout << v << "\t" ;
			if (res.find(v) != res.end()) {
				cur_recall++;
			}
		}
		//cout << endl;
		total_recall += cur_recall;
	}
	
	std::cout << "total_recall = " << total_recall << " " << "num_queries = " <<  num_queries << " recall_at " << recall_at << endl;
	return total_recall / (num_queries) * (100.0 / recall_at);
}

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	auto        val = stat(name.c_str(), &buffer);
	std::cout << " Stat(" << name.c_str() << ") returned: " << val
		<< std::endl;
	return (val == 0);
}
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

/*Helper function*/
void cached_ifstream :: open(const std::string& filename, uint64_t cacheSize) {
	this->cur_off = 0;
	reader.open(filename, std::ios::binary | std::ios::ate);
	fsize = reader.tellg();
	reader.seekg(0, std::ios::beg);
	assert(reader.is_open());
	assert(cacheSize > 0);
	cacheSize = (std::min)(cacheSize, fsize);
	this->cache_size = cacheSize;
	cache_buf = new char[cacheSize];
	reader.read(cache_buf, cacheSize);
	cout << "Opened: " << filename.c_str() << ", size: " << fsize
		<< ", cache_size: " << cacheSize << std::endl;
}

size_t cached_ifstream :: get_file_size() {
	return fsize;
}
void cached_ifstream :: read(char* read_buf, uint64_t n_bytes) {
	assert(cache_buf != nullptr);
	assert(read_buf != nullptr);
	if (n_bytes <= (cache_size - cur_off)) {
		// case 1: cache contains all data
		memcpy(read_buf, cache_buf + cur_off, n_bytes);
		cur_off += n_bytes;
	} else {
		// case 2: cache contains some data
		uint64_t cached_bytes = cache_size - cur_off;
		if (n_bytes - cached_bytes > fsize - reader.tellg()) {
			std::stringstream stream;
			stream << "Reading beyond end of file" << std::endl;
			stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes
				<< " fsize: " << fsize << " current pos:" << reader.tellg()
				<< std::endl;
			cout << stream.str() << std::endl;
			exit(1);
		}
		memcpy(read_buf, cache_buf + cur_off, cached_bytes);

		reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
		cur_off = cache_size;

		uint64_t size_left = fsize - reader.tellg();

		if (size_left >= cache_size) {
			reader.read(cache_buf, cache_size);
			cur_off = 0;
		}

	}
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

int main(int argc, char** argv)
{

	if (argc < 6) {
		cerr << "Too few parameters! " << argv[0] << " " << "<path with file prefix to the director with index files > \
		<query file> <GroundTruth File> <NumQueries> <recall parameter k>"  << endl;
		exit(1);
	}
	cout << "Sizeof result_ann_t : " << sizeof(result_ann_t) << endl;
	/*

	1. Directory of INdex Files
	2. Query file
	3. GroundTruth File
	4. Number of Queries
	5. 
	*/ 
	bang_load<uint8_t>(argv[1]);

	// load the queries
	uint8_t* queriesFP = NULL;
	int numQueries = atoi(argv[4]);
	
	string queryPointsFP_file = string(argv[2]);
	ifstream in4(queryPointsFP_file, std::ios::binary);
	if(!in4.is_open()){
		printf("Error.. Could not open the Query File: %s\n", queryPointsFP_file.c_str());
		return -1;
	}

	in4.seekg(4);
	int Dim = 0;
	in4.read((char*)&Dim, sizeof(int));
	
	queriesFP = (uint8_t*) malloc(sizeof(uint8_t)* numQueries * Dim);	// full floating point coordinates of queries

	if (NULL == queriesFP)
	{
		printf("Error.. Malloc failed for queriesFP");
		return -1;
	}

	in4.read((char*)queriesFP,sizeof(uint8_t)*Dim*numQueries);
	in4.close();

	int recall_param = atoi(argv[5]);
	int nWLLen = 10;
	do
	{
		
		cout << "Enter value of WorkList Length" << endl;
		cin >> nWLLen;

		result_ann_t*  nearestNeighbours = (result_ann_t*)malloc(sizeof(result_ann_t) * recall_param * numQueries);
		float*  nearestNeighbours_dist = (float*)malloc(sizeof(float) * recall_param * numQueries);
		vector<vector<result_ann_t>> final_bestL1;	// Per query vector to store the visited parent and its distance to query point
		final_bestL1.resize(numQueries);

		bang_set_searchparams(recall_param, nWLLen);
		bang_query<uint8_t>(queriesFP,  numQueries, nearestNeighbours,  nearestNeighbours_dist ) ;

		// compute recall
		unsigned numCPUthreads = 64;
		omp_set_num_threads(numCPUthreads);
		#pragma omp parallel
		{
			int CPUthreadno =  omp_get_thread_num();
			vector<result_ann_t> query_best;

			for(unsigned ii=CPUthreadno; ii < numQueries; ii = ii + numCPUthreads)
			{
				query_best.clear();
				for(unsigned jj=0; jj < recall_param; ++jj)
				{
					//query_best.push_back(nearestNeighbours[ ( numQueries * jj) + ii]);
					query_best.push_back(nearestNeighbours[ ( recall_param * ii) + jj]);
				}

				#pragma omp critical
				{
					final_bestL1[ii] = query_best;
	//				printf("final_bestL1[%d] size = %lu \n", ii, final_bestL1[ii].size());
				}
			}		
		}
		// Computing the recall

		unsigned*         gt_ids = nullptr;
		float*            gt_dists = nullptr;
		size_t            gt_num, gt_dim;
		std::vector<uint64_t> Lvec;
		bool calc_recall_flag = false;

		uint64_t curL = nWLLen;
		if (curL >= recall_param)
			Lvec.push_back(curL);

		// Like DiskANN o/p we wannted to run with multiple L's in a single invocation of the program
		// So, test_id to each run with a specific L value. But, currently, we have only one L value
		// i.e. Lvec.size() == 1
		if (Lvec.size() == 0) {
			std::cout << "No valid Lsearch found. Lsearch must be at least recall_at."
				<< std::endl;
			exit(1);
		}
		if (file_exists(argv[3])) {
			load_truthset(argv[3], gt_ids, gt_dists, gt_num, gt_dim);
			calc_recall_flag = true;
		}
		std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
		std::cout.precision(2);
		std::string         recall_string = "Recall@" + std::to_string(recall_param);
		cout << "Ls\t" << recall_string  << endl;
		std::vector<std::vector<result_ann_t> > query_result_ids(Lvec.size());

		unsigned total_size=0;

		for (unsigned test_id = 0; test_id < Lvec.size(); test_id++) {
			unsigned L1 = Lvec[test_id];
			query_result_ids[test_id].resize(recall_param * numQueries);

			for(unsigned ii = 0; ii < numQueries; ++ii) {
				//total_size += final_bestL1[ii].size();
				for(unsigned jj = 0; jj < recall_param; ++jj) {
					query_result_ids[test_id][ii*recall_param+jj] = final_bestL1[ii][jj];
				}
			}
			float recall = 0;
			if (calc_recall_flag)
				recall = calculate_recall(numQueries, gt_ids, gt_dists, gt_dim, query_result_ids[test_id].data(), recall_param, recall_param);
			cout << L1 << "\t" << recall << endl;
		}

		free(nearestNeighbours);
		nearestNeighbours = NULL;
		free(nearestNeighbours_dist);
		nearestNeighbours_dist = NULL;
		char c = 'n';
		cout << "Try Next run ? [y|n]" << endl;
		cin >> c;
		if (c !='y')
			break;
	}while (1);

	free(queriesFP);
}



