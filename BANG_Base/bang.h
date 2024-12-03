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

#ifndef BANG_H_
#define BANG_H_
typedef unsigned long result_ann_t ; // big-ann-benchmarks requires the final ANNs to be returned as int64_t

// Type of Similarity distnace measure
typedef enum _DistFunc
{
	ENUM_DIST_L2 = 0, // Euclidean Distance
	ENUM_DIST_MIPS, // Max Inner Product Search
} DistFunc;
#define MIPS_EXTRA_DIM (1) // To transform MIPS to L2 distance caluclation, extra dim is added to base dataset adn query
                          // The index file o/p from DiskANN already has 1 DIM added to dataset. We(BANG) add 1 DIM to the query 
                          // at rum time


/*! @brief Load the graph index, compressed vectors etc into CPU/GPU memory.
*
* The graph index, compressed vectors has to be generated using DiskANN.
* Search can be performed bang_query() .
*
* @param[in] indexfile_path_prefix Absolute path location where DiskANN generated files are present. (including the file prefix)

*/
template<typename T>
void bang_load( char* indexfile_path_prefix);

extern "C" void bang_load_c( char* indexfile_path_prefix);


void bang_set_searchparams(int recall, 
                            int worklist_length,
                            DistFunc nDistFunc=ENUM_DIST_L2);

extern "C"  void bang_set_searchparams_c(int recall, int worklist_length, DistFunc nDistFunc=ENUM_DIST_L2);

/*! @brief Runs search queries on the laoded index..
*
* The graph index, compressed vectors has to be generated using DiskANN.
* Search can be performed bang_query() .
*
* @param[in] query_file Absolute path of the query file in bin format.
* @param[in] groundtruth_file Absolute path of the groundtruth file in bin format (generated usign DiskANN).
* @param[in] num_queries Number of queries to be used for the search.
* @param[in] recal_param k-recall@k.

*/

template<typename T>
void bang_query(raft::device_resources handle, T* query_array, 
                    int num_queries, 
                    result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist );

extern "C" void bang_query_c(raft::device_resources handle, uint8_t* query_array, 
                    int num_queries, 
                    result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist );


void bang_unload();

extern "C" void bang_unload_c( );
#endif //BANG_H_
