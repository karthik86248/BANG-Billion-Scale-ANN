# BANG : Billion-Scale Approximate Nearest Neighbor Search using a Single GPU

Efficient Approximate Nearest Neighbour Search using GPU. We have three variants of the implementation :
* BANG Base : The graph is stored on the host RAM, PQ compressed vectors on GPU.
* BANG In-memory : The graph and PQ compressed vectors, both are stored on GPU.
* BANG Exact-distance : The graph is stored on GPU. PQ compressed vectors are not used. Distance computations are performed using the base dataset vectors.

Billion scale datasets can be used with BANG Base only.

The source code for each variant is present in the respective folders.

## Prerequisities
* Sufficient Host RAM to store the graph per the dataset (Highest being 640 GB for DEEP1B)
* NVIDIA A100 80GB GPU card
* CUDA version >= 11.8
* gcc and g++ 11.0 or higher (C++11 support)
* Boost C++ libraries (https://www.boost.org/) version >=1.74
* DiskANN (follow the instructions provided in https://github.com/microsoft/DiskANN). Note: Don't use the latest version. Use  Version 0.1 or 0.2. The generated binary files are different in the latest version and BANG_Exact code is not updated to consume the latest format from DiskANN.

## Dataset repositories
SIFT and GIST datasets can be downloaded from http://corpus-texmex.irisa.fr/

GLOVE200 and NYTIMES can be downloaded from https://github.com/erikbern/ann-benchmarks/blob/master/README.md

MNIST8M can be downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist8m

DEEP100M is to be cut out from DEEP1B. Take first 100M points. https://big-ann-benchmarks.com/


Note: For MNIST, zero rows must be removed from the base file.

Note: For GIST1M, there were only 1000 queries. Therefore 1000 queries were repeated 10 times to give queries file of 10,000  queries.

The rest of the steps are specific to BANG_Exact variant
## Graph Generation
* Download the base dataset from the respective dataset repository.
* * Generate the graph using the *build_disk_index* utility.
* * Generate the groundtruth using the *compute_groundtruth* utility.

```
e.g../build_disk_index --data_type uint8 --dist_fn l2 --data_path /mnt/hdd_volume/datasets/sift1b/bigann_base.bin --index_path_prefix sift1b_index -R 64 -L 200 -B 70 -M 48
```
* The o/p generates several files. Below are required by BANG
```
<X>_index_disk.index -> The Vamana graph. Convet this file to bin format using the index_to_binary_graph.py in utils folder.
<X>_index_pq_compressed.bin -> Compressed vectors
<X>_index_pq_pivots.bin 
<X>_index_pq_pivots.bin_centroid.bin
<X>_index_pq_pivots.bin_chunk_offsets.bin
```
Also, Note the number of chunks (PQ Compression) used for generateion of compressed vectors. The number of chunks is dependent in the amount of RAM specified in the -M parameter of build_disk_index.
refer: (https://github.com/karthik86248/BANG-Billion-Scale-ANN/blob/main/BANG_Base/ReadMe.pdf) for DiskANN usave.
## Code Build
Run the below script from the root folder. The supported datasets are captured as #defines in the code/parANN_skeleton.h.
For each dataset, a separate compilation is required.
```
./compile.sh <dataset name> <worklist length> <no of chunks used in PQ compression>
```
## ANN Search on the generated graph
Note down the location of the DiskANN Generated files. Update the respective file locations in the sift100m64.sh. 
An example of bang search is shown below. 
```
./bang_exact //mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_pivots.bin  /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_compressed.bin /mnt/karthik_hdd_4tb/sift100m128/sift100m_graph.bin  /mnt/karthik_hdd_4tb/sift100m128/sift100m_query.bin /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_pivots.bin_chunk_offsets.bin /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_pivots.bin_centroid.bin  /mnt/karthik_hdd_4tb/sift100m128/sift100m_gndtruth.bin 10000 1 256 512 256 10 64 0

```
The sift100m64.sh can be readily executed for SIFT100m dataset with 64 chunks in PQ compressed vectors.
The results are captured in output.txt



