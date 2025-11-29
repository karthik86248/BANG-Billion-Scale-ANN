// ============================================================================
// FUSED KERNEL: Combines neighbor_filtering + compute_neighborDist + bitonic sort/merge
// ============================================================================
#define BF_ENTRIES 399887U
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned); 
#define R 64
#define SIZEPARENTLIST  (1+1)
#ifndef BITONIC_SORT_SIZE
#define BITONIC_SORT_SIZE 64 
#endif

// Hash function for bloom filter
__device__ __forceinline__ unsigned hashFn1_fused(unsigned x) {
    uint64_t hash = 0xcbf29ce4;
    hash = (hash ^ (x & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;
    return hash % BF_ENTRIES;
}

// Device function: Bitonic sort for 64 elements
__device__ __forceinline__ void bitonicSortFused(
    float* shm_dist, 
    unsigned* shm_idx,
    unsigned length,
    unsigned tid,
    unsigned blockSize)
{
    // Pad remaining elements with MAX values (Up to 64)
    for (unsigned i = length + tid; i < BITONIC_SORT_SIZE; i += blockSize) {
        shm_dist[i] = 3.402823466e+38f;
        shm_idx[i] = 0xFFFFFFFF;
    }
    __syncthreads();
    
    // 6 stages for 64 elements
    #pragma unroll
    for (int sort_len = 0; sort_len < 6; sort_len++) {
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

__global__ void fused_search_kernel(
    uint8_t* __restrict__ d_pIndex,
    datatype_t* __restrict__ d_queriesFP,
    bool* __restrict__ d_processed_bit_vec,
    unsigned* __restrict__ d_parents,
    unsigned* __restrict__ d_BestLSets,
    float* __restrict__ d_BestLSetsDist,
    bool* __restrict__ d_BestLSets_visited,
    unsigned* __restrict__ d_BestLSets_count,
    unsigned iter,
    bool* __restrict__ d_nextIter,
    unsigned* __restrict__ d_numNeighbors_query)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    
    // Shared Memory
    __shared__ unsigned shm_neighbors[R + 1];
    __shared__ float shm_neighborsDist[R + 1]; 
    __shared__ unsigned shm_numNeighbors;
    
    // Merge buffers
    __shared__ float shm_currBestLSetsDist[L];
    __shared__ float shm_BestLSetsDist[L];
    __shared__ unsigned shm_pos[(R + 1) + L]; 
    __shared__ unsigned shm_BestLSets[L];
    __shared__ bool shm_BestLSets_visited[L];
    __shared__ unsigned nbrsBound;
    
    if (tid == 0) shm_numNeighbors = 0;
    __syncthreads();
    
    // =========================================================================
    // PHASE 1: NEIGHBOR FILTERING 
    // =========================================================================
    if (d_parents[queryID * SIZEPARENTLIST] == 0 && iter > 1) return;
    
    unsigned offset_bit_vec = queryID * BF_MEMORY;
    bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;
    unsigned long long parentID;
    
    if (iter == 1) {
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
    
    unsigned* bound = (unsigned*)(d_pIndex + ((unsigned long long)INDEX_ENTRY_LEN * parentID) + D * sizeof(datatype_t));
    unsigned numParentNeighbors = *bound;
    
    for (unsigned ii = tid; ii < numParentNeighbors; ii += blockDim.x) {
        unsigned nbr = *(bound + 1 + ii);
        unsigned hashVal = hashFn1_fused(nbr);
        if (!d_processed_bit_vec_start[hashVal]) {
            d_processed_bit_vec_start[hashVal] = true;
            unsigned old = atomicAdd(&shm_numNeighbors, 1);
            if (old < R + 1) shm_neighbors[old] = nbr;
        }
    }
    __syncthreads();
    
    unsigned numNeighbors = min(shm_numNeighbors, (unsigned)(R + 1));
    if (d_numNeighbors_query != NULL && tid == 0) d_numNeighbors_query[queryID] = numNeighbors;
    
    // =========================================================================
    // PHASE 2: DISTANCE COMPUTATION 
    // =========================================================================
    for (unsigned i = tid; i < R + 1; i += blockDim.x) shm_neighborsDist[i] = 3.402823466e+38f;
    __syncthreads();
    
    if (numNeighbors > 0) {
        datatype_t* d_queriesFP_start = d_queriesFP + (queryID * D);
        #define THREADS_PER_NBR 8
        for (unsigned j = tid / THREADS_PER_NBR; j < numNeighbors; j += blockDim.x / THREADS_PER_NBR) {
            unsigned myNeighbor = shm_neighbors[j];
            datatype_t* pBase = (datatype_t*)(d_pIndex + ((unsigned long long)myNeighbor * INDEX_ENTRY_LEN));
            float sum = 0.0f;
            for (unsigned i = tid % THREADS_PER_NBR; i < D; i += THREADS_PER_NBR) {
                float diff = (float)pBase[i] - (float)d_queriesFP_start[i];
                sum += diff * diff;
            }
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) sum += __shfl_down_sync(0xFF, sum, offset);
            if (tid % THREADS_PER_NBR == 0) shm_neighborsDist[j] = sum;
        }
    }
    __syncthreads();
    
    // =========================================================================
    // PHASE 3: BITONIC SORT + BINARY SEARCH INSERTION 
    // =========================================================================
    __shared__ unsigned shm_sortIdx[R + 1];
    for (unsigned i = tid; i < R + 1; i += blockDim.x) {
        shm_sortIdx[i] = (i < numNeighbors) ? shm_neighbors[i] : 0xFFFFFFFF;
    }
    __syncthreads();
    
    // 1. Bitonic Sort on first 64 (Bitonic Sort handles padding if numNeighbors < 64)
    if (numNeighbors > 1) {
        unsigned sortSize = (numNeighbors < BITONIC_SORT_SIZE) ? numNeighbors : BITONIC_SORT_SIZE;
        bitonicSortFused(shm_neighborsDist, shm_sortIdx, sortSize, tid, blockDim.x);
    }
    
    // 2. Insert 65th element (Overflow) if it exists
    if (numNeighbors > BITONIC_SORT_SIZE) {
        __syncthreads(); // Wait for bitonic to finish
        
        // Single thread performs insertion to ensure safety
        if (tid == 0) {
            float valToInsert = shm_neighborsDist[BITONIC_SORT_SIZE]; // Element at index 64
            unsigned idxToInsert = shm_sortIdx[BITONIC_SORT_SIZE];
            
            // --- Binary Search (Upper Bound) ---
            // Find first position where dist > valToInsert
            int left = 0;
            int right = BITONIC_SORT_SIZE; // 64
            
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (shm_neighborsDist[mid] <= valToInsert) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            
            int insertPos = left;
            
            // --- Shift and Insert ---
            // Move elements from [insertPos ... 63] to [insertPos+1 ... 64]
            // We iterate backwards to prevent overwriting
            for (int k = BITONIC_SORT_SIZE; k > insertPos; k--) {
                shm_neighborsDist[k] = shm_neighborsDist[k-1];
                shm_sortIdx[k] = shm_sortIdx[k-1];
            }
            
            // Place the new element
            shm_neighborsDist[insertPos] = valToInsert;
            shm_sortIdx[insertPos] = idxToInsert;
        }
        __syncthreads(); 
    }

    // =========================================================================
    // MERGE LOGIC (Standard merge with sorted lists)
    // =========================================================================
    unsigned Best_L_Set_size = 0;
    unsigned newBest_L_Set_size = 0;
    
    if (numNeighbors > 0) {
        if (iter == 1) {
            nbrsBound = min(numNeighbors, (unsigned)L);
            for (unsigned ii = tid; ii < nbrsBound; ii += blockDim.x) {
                unsigned nbr = shm_sortIdx[ii];
                d_BestLSets[queryID * L + ii] = nbr;
                d_BestLSetsDist[queryID * L + ii] = shm_neighborsDist[ii];
                d_BestLSets_visited[queryID * L + ii] = (nbr == MEDOID);
            }
            __syncthreads();
            if (tid == 0) d_BestLSets_count[queryID] = nbrsBound;
            newBest_L_Set_size = nbrsBound;
        }
        else {
            Best_L_Set_size = d_BestLSets_count[queryID];
            float maxBestLSetDist = d_BestLSetsDist[L * queryID + Best_L_Set_size - 1];
            
            if (tid == 0) {
                unsigned bound = min((unsigned)L, numNeighbors);
                for (nbrsBound = 0; nbrsBound < bound; ++nbrsBound) {
                    if (shm_neighborsDist[nbrsBound] >= maxBestLSetDist) break;
                }
                nbrsBound = max(nbrsBound, min((unsigned)(L - Best_L_Set_size), numNeighbors));
            }
            __syncthreads();
            
            newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, (unsigned)L);
            if (tid == 0) d_BestLSets_count[queryID] = newBest_L_Set_size;
            
            for (unsigned i = tid; i < Best_L_Set_size; i += blockDim.x) shm_currBestLSetsDist[i] = d_BestLSetsDist[L * queryID + i];
            __syncthreads();
            
            if (tid < nbrsBound) shm_pos[tid] = lower_bound_d(shm_currBestLSetsDist, 0, Best_L_Set_size, shm_neighborsDist[tid]) + tid;
            if (tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size)) {
                unsigned localIdx = tid - nbrsBound;
                shm_pos[tid] = upper_bound_d(shm_neighborsDist, 0, nbrsBound, shm_currBestLSetsDist[localIdx]) + localIdx;
            }
            __syncthreads();
            
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
        if (parentIndex == 0) d_parents[queryID * SIZEPARENTLIST] = 0;
    }
}