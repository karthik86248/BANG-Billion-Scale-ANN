// ============================================================================
// FUSED KERNEL: Combines neighbor_filtering + compute_neighborDist + bitonic sort/merge
// 
// Benefits:
// 1. Eliminates 160 kernel launches (80 iterations × 2 extra kernels)
// 2. Neighbors stay in shared memory (no global memory round-trip)
// 3. Distances computed directly into shared memory
// 
// Replace the THREE separate kernel launches with ONE fused kernel launch
// ============================================================================
#define R 64
#define SIZEPARENTLIST  (1+1)
#ifndef BITONIC_SORT_SIZE
#define BITONIC_SORT_SIZE 128
#endif
#define BF_ENTRIES 399887U 
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned); // 4-byte mem aligned size for actual allocation

// Hash function for bloom filter (same as original)
__device__ __forceinline__ unsigned hashFn1_fused(unsigned x) {
    uint64_t hash = 0xcbf29ce4;
    hash = (hash ^ (x & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;
    return hash % BF_ENTRIES;
}

// Device function: Bitonic sort for (distance, index) pairs
__device__ __forceinline__ void bitonicSortFused(
    float* shm_dist, 
    unsigned* shm_idx,
    unsigned length,
    unsigned tid,
    unsigned blockSize)
{
    // Pad remaining elements with MAX values
    for (unsigned i = length + tid; i < BITONIC_SORT_SIZE; i += blockSize) {
        shm_dist[i] = 3.402823466e+38f;
        shm_idx[i] = 0xFFFFFFFF;
    }
    __syncthreads();
    
    // Bitonic sort: 7 stages for 128 elements
    #pragma unroll
    for (int sort_len = 0; sort_len < 7; sort_len++) {
        for (int bitonic_len = sort_len; bitonic_len >= 0; bitonic_len--) {
            int bitonic_len_val = 1 << bitonic_len;
            int bitonic_len_mask = ~(bitonic_len_val - 1);
            
            for (unsigned t = tid; t < BITONIC_SORT_SIZE / 2; t += blockSize) {
                int ind = t + (t & bitonic_len_mask);
                int partner = ind + bitonic_len_val;
                
                bool descending = (t >> sort_len) & 0x1;
                bool shouldSwap = (shm_dist[ind] > shm_dist[partner]) != descending;
                
                if (shouldSwap) {
                    float tmpDist = shm_dist[ind];
                    shm_dist[ind] = shm_dist[partner];
                    shm_dist[partner] = tmpDist;
                    
                    unsigned tmpIdx = shm_idx[ind];
                    shm_idx[ind] = shm_idx[partner];
                    shm_idx[partner] = tmpIdx;
                }
            }
            __syncthreads();
        }
    }
}

/**
 * FUSED KERNEL: One iteration of the search algorithm
 * Combines: neighbor_filtering + compute_neighborDist_par + compute_BestLSets_bitonic
 * 
 * Launch with: fused_search_kernel<<<numQueries, 256, 0, stream>>>
 * 
 * IMPORTANT: Host must set d_nextIter to false before launching!
 */
__global__ void fused_search_kernel(
    // Graph and query data
    uint8_t* __restrict__ d_pIndex,
    datatype_t* __restrict__ d_queriesFP,
    
    // Bloom filter for visited tracking
    bool* __restrict__ d_processed_bit_vec,
    
    // Parent tracking
    unsigned* __restrict__ d_parents,
    
    // BestLSets (candidate set)
    unsigned* __restrict__ d_BestLSets,
    float* __restrict__ d_BestLSetsDist,
    bool* __restrict__ d_BestLSets_visited,
    unsigned* __restrict__ d_BestLSets_count,
    
    // Iteration control
    unsigned iter,
    bool* __restrict__ d_nextIter,
    
    // For statistics (optional, can pass NULL after debugging)
    unsigned* __restrict__ d_numNeighbors_query)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    
    // =========================================================================
    // SHARED MEMORY ALLOCATION
    // =========================================================================
    
    // For neighbor filtering & distance computation
    __shared__ unsigned shm_neighbors[R + 1];
    __shared__ float shm_neighborsDist[BITONIC_SORT_SIZE];
    __shared__ unsigned shm_numNeighbors;
    
    // For merge operation
    __shared__ float shm_currBestLSetsDist[L];
    __shared__ float shm_BestLSetsDist[L];
    __shared__ unsigned shm_pos[BITONIC_SORT_SIZE + L];
    __shared__ unsigned shm_BestLSets[L];
    __shared__ bool shm_BestLSets_visited[L];
    __shared__ unsigned nbrsBound;
    
    // Initialize neighbor count
    if (tid == 0) {
        shm_numNeighbors = 0;
    }
    __syncthreads();
    
    // =========================================================================
    // PHASE 1: NEIGHBOR FILTERING (from neighbor_filtering_new)
    // =========================================================================
    
    // Check if this query has a parent to process
    if (d_parents[queryID * SIZEPARENTLIST] == 0 && iter > 1) {
        // No parent, this query is done
        return;
    }
    
    unsigned offset_bit_vec = queryID * BF_MEMORY;
    bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;
    unsigned long long parentID;
    
    if (iter == 1) {
        // First iteration: add MEDOID
        parentID = MEDOID;
        if (tid == 0) {
            if (!d_processed_bit_vec_start[hashFn1_fused(MEDOID)]) {
                d_processed_bit_vec_start[hashFn1_fused(MEDOID)] = true;
                shm_neighbors[0] = MEDOID;
                shm_numNeighbors = 1;
            }
        }
        __syncthreads();
    } else {
        parentID = d_parents[queryID * SIZEPARENTLIST + 1];
    }
    
    // Get neighbors of parent from graph
    unsigned* bound = (unsigned*)(d_pIndex + ((unsigned long long)INDEX_ENTRY_LEN * parentID) + D * sizeof(datatype_t));
    unsigned numParentNeighbors = *bound;
    
    // Filter neighbors through bloom filter
    for (unsigned ii = tid; ii < numParentNeighbors; ii += blockDim.x) {
        unsigned nbr = *(bound + 1 + ii);
        unsigned hashVal = hashFn1_fused(nbr);
        
        if (!d_processed_bit_vec_start[hashVal]) {
            d_processed_bit_vec_start[hashVal] = true;
            unsigned old = atomicAdd(&shm_numNeighbors, 1);
            if (old < R + 1) {
                shm_neighbors[old] = nbr;
            }
        }
    }
    __syncthreads();
    
    unsigned numNeighbors = min(shm_numNeighbors, (unsigned)(R + 1));
    
    // Store for statistics (optional)
    if (d_numNeighbors_query != NULL && tid == 0) {
        d_numNeighbors_query[queryID] = numNeighbors;
    }
    
    // =========================================================================
    // PHASE 2: DISTANCE COMPUTATION (from compute_neighborDist_par)
    // Neighbors are already in shared memory - no global memory read needed!
    // =========================================================================
    
    // Initialize distances to MAX
    for (unsigned i = tid; i < BITONIC_SORT_SIZE; i += blockDim.x) {
        shm_neighborsDist[i] = 3.402823466e+38f;
    }
    __syncthreads();
    
    if (numNeighbors > 0) {
        datatype_t* d_queriesFP_start = d_queriesFP + (queryID * D);
        
        // Process neighbors - 8 threads per neighbor for reduction
        #define THREADS_PER_NBR 8
        
        for (unsigned j = tid / THREADS_PER_NBR; j < numNeighbors; j += blockDim.x / THREADS_PER_NBR) {
            unsigned myNeighbor = shm_neighbors[j];
            datatype_t* pBase = (datatype_t*)(d_pIndex + ((unsigned long long)myNeighbor * INDEX_ENTRY_LEN));
            
            float sum = 0.0f;
            
            // Each thread computes partial sum
            for (unsigned i = tid % THREADS_PER_NBR; i < D; i += THREADS_PER_NBR) {
                float diff = (float)pBase[i] - (float)d_queriesFP_start[i];
                sum += diff * diff;
            }
            
            // Warp shuffle reduction for 8 threads
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFF, sum, offset);
            }
            
            // First thread in group writes result
            if (tid % THREADS_PER_NBR == 0) {
                shm_neighborsDist[j] = sum;
            }
        }
    }
    __syncthreads();
    
    // =========================================================================
    // PHASE 3: BITONIC SORT + MERGE (from compute_BestLSets_bitonic)
    // Distances are already in shared memory - no global memory read needed!
    // =========================================================================
    
    // Copy neighbor IDs for sorting (distances already in shm_neighborsDist)
    __shared__ unsigned shm_sortIdx[BITONIC_SORT_SIZE];
    for (unsigned i = tid; i < BITONIC_SORT_SIZE; i += blockDim.x) {
        shm_sortIdx[i] = (i < numNeighbors) ? shm_neighbors[i] : 0xFFFFFFFF;
    }
    __syncthreads();
    
    // Bitonic sort
    if (numNeighbors > 1) {
        bitonicSortFused(shm_neighborsDist, shm_sortIdx, numNeighbors, tid, blockDim.x);
    }
    
    // Now merge with BestLSets
    unsigned Best_L_Set_size = 0;
    unsigned newBest_L_Set_size = 0;
    
    if (numNeighbors > 0) {
        
        if (iter == 1) {
            // First iteration: Initialize BestLSets
            nbrsBound = min(numNeighbors, (unsigned)L);
            
            for (unsigned ii = tid; ii < nbrsBound; ii += blockDim.x) {
                unsigned nbr = shm_sortIdx[ii];
                d_BestLSets[queryID * L + ii] = nbr;
                d_BestLSetsDist[queryID * L + ii] = shm_neighborsDist[ii];
                d_BestLSets_visited[queryID * L + ii] = (nbr == MEDOID);
            }
            __syncthreads();
            
            newBest_L_Set_size = nbrsBound;
            if (tid == 0) {
                d_BestLSets_count[queryID] = nbrsBound;
            }
        }
        else {
            // Subsequent iterations: Merge
            Best_L_Set_size = d_BestLSets_count[queryID];
            float maxBestLSetDist = d_BestLSetsDist[L * queryID + Best_L_Set_size - 1];
            
            if (tid == 0) {
                unsigned bound = min((unsigned)L, numNeighbors);
                for (nbrsBound = 0; nbrsBound < bound; ++nbrsBound) {
                    if (shm_neighborsDist[nbrsBound] >= maxBestLSetDist) {
                        break;
                    }
                }
                nbrsBound = max(nbrsBound, min((unsigned)(L - Best_L_Set_size), numNeighbors));
            }
            __syncthreads();
            
            newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, (unsigned)L);
            
            if (tid == 0) {
                d_BestLSets_count[queryID] = newBest_L_Set_size;
            }
            
            // Load current BestLSetsDist
            for (unsigned i = tid; i < Best_L_Set_size; i += blockDim.x) {
                shm_currBestLSetsDist[i] = d_BestLSetsDist[L * queryID + i];
            }
            __syncthreads();
            
            // Calculate merge positions
            if (tid < nbrsBound) {
                shm_pos[tid] = lower_bound_d(shm_currBestLSetsDist, 0, Best_L_Set_size, shm_neighborsDist[tid]) + tid;
            }
            if (tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size)) {
                unsigned localIdx = tid - nbrsBound;
                shm_pos[tid] = upper_bound_d(shm_neighborsDist, 0, nbrsBound, shm_currBestLSetsDist[localIdx]) + localIdx;
            }
            __syncthreads();
            
            // Write merged results
            if (tid < nbrsBound && shm_pos[tid] < newBest_L_Set_size) {
                shm_BestLSetsDist[shm_pos[tid]] = shm_neighborsDist[tid];
                shm_BestLSets[shm_pos[tid]] = shm_sortIdx[tid];
                shm_BestLSets_visited[shm_pos[tid]] = false;
            }
            if (tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size)) {
                unsigned localIdx = tid - nbrsBound;
                if (shm_pos[tid] < newBest_L_Set_size) {
                    shm_BestLSetsDist[shm_pos[tid]] = shm_currBestLSetsDist[localIdx];
                    shm_BestLSets[shm_pos[tid]] = d_BestLSets[queryID * L + localIdx];
                    shm_BestLSets_visited[shm_pos[tid]] = d_BestLSets_visited[queryID * L + localIdx];
                }
            }
            __syncthreads();
            
            // Copy back to global memory
            for (unsigned i = tid; i < newBest_L_Set_size; i += blockDim.x) {
                d_BestLSetsDist[L * queryID + i] = shm_BestLSetsDist[i];
                d_BestLSets[L * queryID + i] = shm_BestLSets[i];
                d_BestLSets_visited[L * queryID + i] = shm_BestLSets_visited[i];
            }
            __syncthreads();
        }
    }
    
    // =========================================================================
    // PHASE 4: SELECT NEXT PARENT
    // =========================================================================
    
    if (tid == 0) {
        unsigned parentIndex = 0;
        for (unsigned ii = 0; ii < newBest_L_Set_size; ++ii) {
            if (!d_BestLSets_visited[L * queryID + ii]) {
                parentIndex++;
                d_BestLSets_visited[L * queryID + ii] = true;
                d_parents[queryID * SIZEPARENTLIST] = parentIndex;
                d_parents[queryID * SIZEPARENTLIST + parentIndex] = d_BestLSets[L * queryID + ii];
                *d_nextIter = true;
                break;
            }
        }
        
        if (parentIndex == 0) {
            d_parents[queryID * SIZEPARENTLIST] = 0;
        }
    }
}

// ============================================================================
// END OF FUSED KERNEL
// ============================================================================