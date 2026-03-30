#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel for Store Benchmark
// Goal: Measure Bandwidth and Energy of Write operations
// Logic: Grid-Stride Loop for coalesced stores
__global__ void store_kernel(uint64_t* array, uint64_t num_elements, uint64_t stride_elements, uint64_t iterations) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridSize = blockDim.x * gridDim.x;
    
    uint64_t val = tid; // Value to write
    uint64_t* base_ptr = array;

    // Main Loop - Iterations over the full array flush
    // We want to force DRAM writes. Best way is to write DIFFERENT data or to huge array.
    // If array > L2, linear write will flush L2.
    for (uint64_t i = 0; i < iterations; ++i) {
        // Grid-Stride Loop
        // Each thread processes elements spaced by grid size
        // e.g. Thread 0 does 0, 0+Grid, 0+2*Grid...
        // This ensures full coalescing and no collision between threads.
        for (uint64_t idx = tid; idx < num_elements; idx += gridSize) {
             // We can unroll here manually if needed, but compiler is good at linear unrolling
             // Let's do a small unroll block if strictly needed, but simple is better for bandwidth
             // Simple store:
             // array[idx] = val;
             
             // Use PTX to ensure it's a global store and not optimized away
             asm volatile (
                "st.global.u64 [%0], %1;" 
                :: "l"(base_ptr + idx), "l"(val) 
                : "memory"
            );
        }
        // Ensure all writes are visible before next iteration (though for bandwidth, just pushing pulses is fine)
        // __threadfence(); 
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

    // device allocation (no host init needed per spec, but we usually want clear memory)
    uint64_t* d_array;
    CHECK_CUDA(cudaMalloc(&d_array, total_bytes));
    CHECK_CUDA(cudaMemset(d_array, 0, total_bytes));

    // Launch Config
    // Saturate the GPU for bandwidth measurement
    int threads_per_block = 256;
    // Determine grid size: we want enough parallelism to hide latencies
    // A fixed large number of blocks or dependent on array size?
    // For bandwidth tests, usually size/block_size
    // FIXED: Ensure we launch enough blocks to cover the array at least once if possible,
    // or cap it to avoid excessive launch overhead if array is huge.
    // However, for st.cu, each thread has its own loop "idx = (idx + stride*blockDim*gridDim) % num".
    // Wait, the kernel logic was "idx = (idx + stride)".
    // If every thread does "idx += stride", and we have T threads.
    // Thread 0: 0, 4, 8...
    // Thread 1: 1, 5, 9...
    // They collide if not spaced out!
    // Correct strided pattern for bandwidth should be:
    // idx = tid + (i * stride * gridDim * blockDim)?
    // No, standard memory copy style is:
    // for (idx = tid; idx < num; idx += blockDim * gridDim)
    
    // The current kernel uses a persistent thread style:
    // "idx = tid % num; ... idx = (idx + stride)"
    // If stride is small (e.g. 1), Thread 0 does 0, 1, 2...
    // Thread 1 does 1, 2, 3...
    // They confirmably write to the SAME locations constantly.
    // This causes massive contention and L2 write combining, preventing DRAM flush.
    // We should use Grid-Stride Loop pattern to ensure unique writes.
    
    // Changing kernel invocation to match new kernel logic (see below) or keep existing and fix logic?
    // I will fix the Kernel logic in a moment. But first, let's fix the block count to be reasonable.
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    if (blocks > 65535) blocks = 65535; // Cap blocks to reasonable number for persistent threads

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    store_kernel<<<blocks, threads_per_block>>>(d_array, num_elements, stride_elements, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measurement
    CHECK_CUDA(cudaEventRecord(start));
    store_kernel<<<blocks, threads_per_block>>>(d_array, num_elements, stride_elements, iterations);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate Bandwidth
    // New Logic: Each iteration writes the ENTIRE array exactly once.
    // Total Bytes = num_elements * sizeof(uint64_t) * iterations
    double total_write_bytes = (double)num_elements * sizeof(uint64_t) * iterations;
    double gb_per_sec = (total_write_bytes / (milliseconds / 1000.0)) / 1e9;

    std::cout << "Array Size: " << size_mb << " MB" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    CHECK_CUDA(cudaFree(d_array));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
