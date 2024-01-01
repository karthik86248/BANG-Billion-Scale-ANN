#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <string.h>
#include <assert.h>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include "utils/timer.h"
#include "parANN.h"


using namespace std;



int main(int argc, char** argv) {

	if(argc < 16) {
		cerr << "Too few parameters! " << argv[0] << " " << "<Todo: Arg list>..."  << endl;
		exit(1);
	}


	/*Method to perform parallel approximate nearest neighbor search */

	parANN(argc, argv);

	return 0;
}






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








/*Helper function*/
/*
our_results is a 1-D array : <query1 NN Ids>  <query2 NN Ids>
<query x NN Ids> = esactly recall_at entries indicating the node ids of the NNs for that query

intersection of gt and res gives the recall.
Note: gt.size() could be >= recall_at duu to tie breaking
*/

double calculate_recall(unsigned num_queries, unsigned *gold_std,
		float *gs_dist, unsigned dim_gs,
		unsigned *our_results, unsigned dim_or,
		unsigned recall_at) {
	double             total_recall = 0;
	std::set<unsigned> gt, res;

	for (size_t i = 0; i < num_queries; i++) {

		gt.clear();
		res.clear();
		unsigned *gt_vec = gold_std + dim_gs * i;
		unsigned *res_vec = our_results + dim_or * i;
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

#if 0
		if (i == 0)
			std::cout << "gt size = " << gt.size()  << "res size = " << res.size() << std::endl;

		set<unsigned>::iterator iter = gt.begin();
		while(iter != gt.end())
		{
			std:: cout << *(iter++) << "\t";
		}
		std::cout << "\t" << std::endl;


		iter = res.begin();
		while(iter != res.end())
		{
			std:: cout << *(iter++) << "\t";
		}
		std::cout << "\t" << std::endl;
#endif
		unsigned cur_recall = 0;
		for (auto &v : gt) {
			if (res.find(v) != res.end()) {
				cur_recall++;
			}
		}
		total_recall += cur_recall;

	}
	std::cout << "total_recall = " << total_recall << " " << "num_queries = " <<  num_queries << " recall_at " << recall_at << endl;
	return total_recall / (num_queries) * (100.0 / recall_at);
}







