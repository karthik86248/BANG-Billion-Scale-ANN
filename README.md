# ANN-GPU
Efficient Approximate Nearest Neighbour Search using GPUs


This is a standalone GPU parallel implementation of *index_search* from DiskANN.

## Prerequisities
* CUDA version >= 10.0
* gcc and g++ 7.5 or higher (C++11 support)
* Boost library

## Code Build

Run Makefile to build the code.
```
make 
```

## Executing the code

Assuming the executable generated from running ```make``` is called *parANN*. Run the code as below:

```
./parANN path_to_PQ_Table path_to_compressed_vectors path_to_adjacency_list path_to_graph_full_coordinates path_to_query_full_coordinates path_to_chunk_offsets_file path_to_centroids_file path_to_ground_truth_file query_batch_size beamwidth K1_thread_block_size K2_thread_block_size K L_search

```
An example is shown below:

```
./parANN PQ_Table.txt compressed_vectors.txt graph_adjacency_list.txt graph_full_coordinates.txt query_points_full_coords.txt chunk_offsets.bin centroid.bin ground_truth.bin 512 1 64 512 5 20
```

A subset of The input files are expected to be ASCII file.

### File formats of ASCII files

 * PQ_Table.txt 
 
 Each line contains **D** space-separated floating point values (D is the dimension of the data). There are 256 such lines. Each line corresponds to the coordinates of a centroid. 

 * compressed_vectors.txt

 Each line contains the compressed vectors of a point. All entries in a line lie in the range [0,255]. 
 There are chunks number of entries per line and there are as many lines as there are datapoints. For instance, for the SIFT1M dataset, the file would contain 10^6 lines.

 * graph_adjacency_list.txt

 The first line contains the number of points (or nodes).
 The second line contains the list of *medoid ids*, separated by space.
 Line 3 onward stores the adjacency list of the node. Line 3 stores the adjacency list of node 0, Line 4 stores the adjacency list of node 1 and so on.
 In line 3 onward, the first entry is the degree of the node, followed by the adjacency list of the node. For instance, if node 2 of the graph has degree 64, then the line 5 of the file would look as shown below:

```
64 1 54 387 ...
```

 The number of lines equals ```(# datapoints) + 2```.

 * graph_full_coordinates.txt

 Each line contains the full-precision coordinates of a data point. There are **D** entries per line, separated by space.
 There are as many lines as there are datapoints. For the SIFT1M dataset, there are 10^6 lines.
 
  
 * query_points_full_coords.txt 
 
 Each line contains the full-precision coordinates of a query point. There are **D** entries per line, separated by space.
 There are as many lines as there are queries.


## Documentation

A detailed description can be found in [*report.pdf*](report.pdf).
