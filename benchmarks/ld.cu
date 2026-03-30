#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <omp.h>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel for Load Benchmark
// Goal: Measure Latency and Throughput using Pointer Chasing
// Logic: Unrolled pointer chasing loop using inline PTX
__global__ void load_kernel(uint64_t* array, uint64_t num_elements,
                            uint64_t stride_elements, uint64_t num_chunks,
                            uint64_t iterations) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out of bounds if grid is larger than needed
    // However, for bandwidth saturation, each thread needs its own chain.
    // If num_elements is small and we have many threads, they will overlap, which is fine for saturation.
    
    if (tid >= num_chunks) return;

    // Start each thread at a unique index (offset by tid)
    // The initialization ensures A[i] points to specific next elements.
    uint64_t current_idx = tid * stride_elements;
    uint64_t* base_ptr = array;

    // Main loop
    for (uint64_t i = 0; i < iterations; ++i) {
        // Unroll 100 times
        #pragma unroll 100
        for (int j = 0; j < 100; ++j) {
            uint64_t next_idx;
            uint64_t* addr = base_ptr + current_idx;
            
            // Inline PTX for ld.global.u64
            // Loads the value at 'addr' into 'next_idx'
            // %0 is next_idx (output), %1 is addr (input address)
            asm volatile (
                "ld.global.u64 %0, [%1];" 
                : "=l"(next_idx) 
                : "l"(addr)
            );
            
            // Pointer chase: update pointer for next iteration
            current_idx = next_idx;
        }
    }

    // Write back result to prevent dead code elimination (though volatile asm usually suffices)
    // We write to a dummy location or just the start location to finish "usage"
    // Using a conditional write to minimize impact on read measurement
    if (current_idx == 0xFFFFFFFFFFFFFFFFULL) {
        array[tid] = current_idx;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <Array Size in MB> <Stride in Bytes> <Iterations>" << std::endl;
        return 1;
    }

    double size_mb = std::stod(argv[1]);
    uint64_t stride_bytes = std::stoull(argv[2]);
    uint64_t iterations = std::stoull(argv[3]);

    size_t total_bytes = (size_t)(size_mb * 1024 * 1024);
    size_t num_elements = total_bytes / sizeof(uint64_t);
    uint64_t stride_elements = stride_bytes / sizeof(uint64_t);

    if (stride_elements == 0) stride_elements = 1;

    // Initialize host array with deterministic strided pointer chain
    // This destroys spatial locality and ensures true DRAM latency/bandwidth measurement
    // by defeating the L2 cache prefetchers.
    
    printf("Initializing strided random chunk chain (stride=%llu elements, %llu bytes)...\n",
           (unsigned long long)stride_elements,
           (unsigned long long)stride_bytes);

    std::vector<uint64_t> h_array(num_elements, 0);

    // Divide the array into chunks of stride_elements each.
    // Shuffle the chunk ORDER randomly (defeats hardware prefetcher —
    // the prefetcher cannot predict which chunk comes next).
    // Within each chunk, only element 0 is used as the chain link;
    // the rest are filled with 0 (never chased).
    // This gives: random traversal order (like Fisher-Yates) but each
    // pointer hop lands exactly stride_elements apart in the chunk index
    // space, keeping accesses sector-aligned.
    uint64_t num_chunks = num_elements / stride_elements;

    std::vector<uint64_t> chunk_order(num_chunks);
    for (uint64_t i = 0; i < num_chunks; ++i) chunk_order[i] = i;

    // Fisher-Yates shuffle on chunks only — much faster than shuffling
    // all elements (num_chunks = num_elements / stride_elements, e.g.
    // for 256MB at stride=32B: 4M chunks vs 32M elements).
    srand(42);
    for (uint64_t i = num_chunks - 1; i > 0; --i) {
        uint64_t j = (uint64_t)rand() % (i + 1);
        std::swap(chunk_order[i], chunk_order[j]);
    }

    // Link chunks in shuffled order.
    // h_array[chunk_order[i] * stride_elements] = chunk_order[i+1] * stride_elements
    // i.e. the first element of chunk i points to the first element of the next chunk.
    omp_set_num_threads(20);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < num_chunks; ++i) {
        uint64_t this_elem = chunk_order[i] * stride_elements;
        uint64_t next_elem = chunk_order[(i + 1) % num_chunks] * stride_elements;
        h_array[this_elem] = next_elem;
        // Fill rest of this chunk with 0 (these elements are never chased)
        for (uint64_t k = 1; k < stride_elements; ++k) {
            h_array[this_elem + k] = 0;
        }
    }

    printf("Strided chain initialised. Chunks: %llu, chunk_size: %llu elements.\n",
           (unsigned long long)num_chunks,
           (unsigned long long)stride_elements);

    // Device allocation
    uint64_t* d_array;
    CHECK_CUDA(cudaMalloc(&d_array, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_array, h_array.data(), total_bytes, cudaMemcpyHostToDevice));

    // Launch with optimal fixed block count to prevent low-occupancy at small sizes
    // We launch enough blocks to saturate the GPU (e.g. 108 SMs * 32 blocks = ~3500 blocks)
    // regardless of array size.
    int threads_per_block = 256; 
    int blocks = 4096; // Fixed high number to ensure saturation

    // However, if array is small, we need to wrap the index access inside the kernel,
    // otherwise threads > num_elements would just return immediately.
    // The current kernel (ld.cu) performs "if (tid >= num_elements) return;".
    // To fix this for the Power Measurement requirement (Sustained Execution on Small Arrays),
    // we need to supply a 'mask' or ensure the kernel handles wrap-around if we launch more threads than elements.
    
    // For replication strictness, we keep the original kernel logic but scale the grid to match array.
    // Since 'load' is pointer chasing, we can't easily launch more threads than elements 
    // without them chasing the SAME pointers (contention).
    // So for 'ld.cu', we stick to (num_elements/256) but ensure 'iterations' is massive in the script.
    
    if (blocks * threads_per_block > num_chunks) {
        blocks = (num_chunks + threads_per_block - 1) / threads_per_block;
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup: traverse the full pointer chain at least once to pull working set into cache.
    // The chain has num_elements links; the kernel does 100 steps per outer iteration.
    uint64_t active_threads_for_warmup = (uint64_t)(blocks * threads_per_block);
    if (active_threads_for_warmup > num_elements) active_threads_for_warmup = num_elements;
    uint64_t warmup_iters = std::max((uint64_t)1,
        num_chunks / (active_threads_for_warmup * 100));
    load_kernel<<<blocks, threads_per_block>>>(d_array, num_elements,
        stride_elements, num_chunks, warmup_iters);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measurement
    CHECK_CUDA(cudaEventRecord(start));
    load_kernel<<<blocks, threads_per_block>>>(d_array, num_elements,
        stride_elements, num_chunks, iterations);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate Bandwidth
    // Each unrolled iteration does 100 loads.
    // Total Ops = Blocks * Threads * Iterations * 100
    // are we measuring Latency (ps) or Throughput (GB/s)?
    // Throughput formula: Total Data Transferred / Time
    // Data per thread = Iterations * 100 * sizeof(uint64_t)
    // Active threads = blocks * threads_per_block (clamped to num_elements in kernel)
    
    size_t active_threads = (size_t)(blocks * threads_per_block);
    if (active_threads > num_chunks) active_threads = num_chunks;
    // Each thread does iterations * 100 pointer hops.
    // Each hop fetches stride_bytes (one sector = 32 bytes for stride=32B).
    double total_data_bytes = (double)active_threads * iterations * 100 * stride_bytes;
    double gb_per_sec = (total_data_bytes / (milliseconds / 1000.0)) / 1e9;

    std::cout << "Stride Bytes: " << stride_bytes << std::endl;
    std::cout << "Array Size: " << size_mb << " MB" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    CHECK_CUDA(cudaFree(d_array));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
