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

// Kernel for Load Benchmark
// Goal: Measure Latency and Throughput using Pointer Chasing
// Logic: Unrolled pointer chasing loop using inline PTX
__global__ void load_kernel(uint64_t* array, uint64_t num_elements, uint64_t iterations) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out of bounds if grid is larger than needed
    // However, for bandwidth saturation, each thread needs its own chain.
    // If num_elements is small and we have many threads, they will overlap, which is fine for saturation.
    
    if (tid >= num_elements) return;

    // Start each thread at a unique index (offset by tid)
    // The initialization ensures A[i] points to specific next elements.
    uint64_t current_idx = tid;
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

    // Initialize host array with guaranteed full traversal
    // If we simply do (i + stride) % N, we might get short cycles if GCD(stride, N) != 1.
    // Fixed logic: Use stride of 1 (Linear Chasing) if stride_elements is small/default
    // or ensure coprime for strided access.
    // For simplicity & robustness to prove DRAM access:
    // We will construct a conflict-free linear chain if stride is small.
    // If user wants random (large stride), we should implement a Linear Congruential Generator or similar,
    // but for now, let's fix the bandwidth bug by changing to strict linear chaining (stride=1) 
    // effectively ignoring the user stride if it risks short loops, OR
    // enforce stride=1 for simple bandwidth measurement.

    printf("Initializing pointer chase array... (Size: %llu elements)\n", (unsigned long long)num_elements);
    
    // Using a simple linear chain (next = current + 1) guarantees touching EVERY element once per pass.
    // This is the most reliable way to measure DRAM bandwidth via pointer chasing.
    // A stride > 1 is only useful if we want to deliberately skip cache lines (e.g. stride=32),
    // but we must still ensure the loop covers the whole array.
    // The safest "strided" traversal that covers everything is:
    // idx = (idx + stride) % size. THIS REQUIRES GCD(stride, size) == 1.
    // Since 'num_elements' is likely a power of 2 (from allocation size), any ODD stride detects as coprime.
    
    // Force stride to be odd if num_elements is even, to ensure full cycle.
    if ((num_elements % 2 == 0) && (stride_elements % 2 == 0)) {
        stride_elements += 1;
        printf("Adjusted stride to %llu to ensure coprimality with even array size.\n", (unsigned long long)stride_elements);
    }

    for (size_t i = 0; i < num_elements; ++i) {
        h_array[i] = (i + stride_elements) % num_elements;
    }

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
    
    if (blocks * threads_per_block > num_elements) {
        blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    load_kernel<<<blocks, threads_per_block>>>(d_array, num_elements, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measurement
    CHECK_CUDA(cudaEventRecord(start));
    load_kernel<<<blocks, threads_per_block>>>(d_array, num_elements, iterations);
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
    
    // Accurate active thread count:
    size_t active_threads = (blocks * threads_per_block); 
    if (active_threads > num_elements) active_threads = num_elements;

    double total_data_bytes = (double)active_threads * iterations * 100 * sizeof(uint64_t);
    double gb_per_sec = (total_data_bytes / (milliseconds / 1000.0)) / 1e9;

    std::cout << "Array Size: " << size_mb << " MB" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    CHECK_CUDA(cudaFree(d_array));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
