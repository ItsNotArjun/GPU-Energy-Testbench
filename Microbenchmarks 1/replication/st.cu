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
// Logic: Unrolled strided stores using inline PTX
__global__ void store_kernel(uint64_t* array, uint64_t num_elements, uint64_t stride_elements, uint64_t iterations) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Start index differs per thread to spread writes across memory controller
    uint64_t idx = tid % num_elements;
    uint64_t val = tid; // Value to write

    uint64_t* base_ptr = array;

    // Main Loop
    for (uint64_t i = 0; i < iterations; ++i) {
        #pragma unroll 100
        for (int j = 0; j < 100; ++j) {
            uint64_t* addr = base_ptr + idx;

            // Inline PTX for st.global.u64
            // Stores 'val' into address 'addr'
            asm volatile (
                "st.global.u64 [%0], %1;" 
                :: "l"(addr), "l"(val) 
                : "memory"
            );

            // Strided access with wrap around
            idx = (idx + stride_elements);
            // Optimization: avoid modulo every inner step if possible, but required for correctness of "stay within bounds"
            // Using a simple check is faster than modulo usually, but modulo is requested logic.
            // "Ensure the index wraps around (idx % size)"
            if (idx >= num_elements) idx -= num_elements; // Fast modulus for linear scan
        }
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
    int blocks = (int)((num_elements + threads_per_block - 1) / threads_per_block);
    
    // Specific constraint: "Launch with optimal block size".
    // Limiting blocks to device capacity if array is huge isn't strictly necessary as GPU schedules them,
    // but having too many might increase overhead.
    // For 1GB array, blocks ~ 10^9 / 256 ~ 4 million. This is fine.

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
    size_t active_threads = blocks * threads_per_block;
    if (active_threads > num_elements) active_threads = num_elements; // Though logic allows wrap around, we only spawn proportional threads
    
    // Each thread does 100 * iterations stores
    double total_write_bytes = (double)active_threads * iterations * 100 * sizeof(uint64_t);
    double gb_per_sec = (total_write_bytes / (milliseconds / 1000.0)) / 1e9;

    std::cout << "Array Size: " << size_mb << " MB" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    CHECK_CUDA(cudaFree(d_array));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
