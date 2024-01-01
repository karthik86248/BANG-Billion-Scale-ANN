# BANG : Billion-Scale Approximate Nearest Neighbor Search using a Single GPU

Efficient Approximate Nearest Neighbour Search using GPU. We have three variants of the implementation :
* BANG_Base : The graph is stored on the host RAM, PQ compressed vectors on GPU.
* BANG_Inmemory : The graph and PQ compressed vectors both are store on GPU.
* BANG_Exactdistance : The graph is stored on GPU. PQ compressed vectors are not used. Distance computations are performed using the base dataset vectors.

Billion scale datasets can be used with Band_Base only.

The source code for each variant is present in the resepctive folders.

## Prerequisities
* Sufficient Host RAM to store the graph per the dataset (Highest being 640 GB for DEEP1B)
* NVIDIA A100 80GB GPU card
* CUDA version >= 11.8
* gcc and g++ 11.0 or higher (C++11 support)
* Boost C++ libraries (https://www.boost.org/) version >=1.74
* DiskANN (follow the instruction in https://github.com/microsoft/DiskANN)

## Dataset repositories
SIFT and GIST datasets can be downloaded from http://corpus-texmex.irisa.fr/

GLOVE200 and NYTIMES can be downloaded from https://github.com/erikbern/ann-benchmarks/blob/master/README.md

MNIST8M can be downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist8m

DEEP100M is to be cut out from DEEP1B. Take first 100M points. https://big-ann-benchmarks.com/


Note: For MNIST, zero rows must be removed from the base file.

Note: For GIST1M, there were only 1000 queries. Therefore 1000 queries were repeated 10 times to give queries file of 10,000  queries.

## Code Build

Run Makefile to build the code.
```
make 
```

## Graph Generation
* Download the base dataset from the respective dataset repository. The base dataset, query vectors and the groundtruth files.
* Generate the graph using the build_disk_index utility.
e.g../build_disk_index --data_type uint8 --dist_fn l2 --data_path /mnt/hdd_volume/datasets/sift1b/bigann_base.bin --index_path_prefix sift1b_index -R 64 -L 200 -B 70 -M 48

* The o/p generates several files. Below are required by BANG
```
<X>_index_disk.index -> The Vamana graph. Convet this file to bin format using the index_to_binary_graph.py in utils folder.
<X>_index_pq_compressed.bin -> Compressed vectors
<X>_index_pq_pivots.bin 
<X>_index_pq_pivots.bin_centroid.bin
<X>_index_pq_pivots.bin_chunk_offsets.bin
```
## ANN Search on the generated graph

```
./bang <<X>_index_pq_pivots.bin> <<X>_index_pq_compressed.bin> <<X>_index_disk.bin> <query vectors file in bin format> <<X>_index_pq_pivots.bin_chunk_offsets.bin> <<X>_index_pq_pivots.bin_centroid.bin> <groundtruth file in bin format> <# of query vectors> <Thread blocl size of compute_parent kernel> <Thread blocl size of populate_pqDist_par kernel> <Thread blocl size of compute_neighborDist_par kernel> <Thread blocl size of neighbor_filtering_new kernel> <recall factor i.e. top-k> <# of OMP threads> <debug flags>

```
An example is shown below:

```
./bang /mnt/ssd_volume/diskANN-working/build/tests/sift1b_index_pq_pivots.bin  /mnt/ssd_volume/diskANN-working/build/tests/sift1b_index_pq_compressed.bin /mnt/ssd_volume/diskANN-working/build/tests/sift1b_index_disk.bin  /mnt/hdd_volume2/sift1b/sift1b_query.bin /mnt/ssd_volume/diskANN-working/build/tests/sift1b_index_pq_pivots.bin_chunk_offsets.bin /mnt/ssd_volume/diskANN-working/build/tests/sift1b_index_pq_pivots.bin_centroid.bin  /mnt/hdd_volume2/sift1b/sift1b_groundtruth.bin 10000 1 256 512 256 10 64 1```




