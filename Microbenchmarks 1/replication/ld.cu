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

    // Host allocation
    std::vector<uint64_t> h_array(num_elements);

    // Initialization: A[i] = (i + stride_index) % total_elements
    // This creates the pointer chase chain
    for (uint64_t i = 0; i < num_elements; ++i) {
        h_array[i] = (i + stride_elements) % num_elements;
    }

    // Device allocation
    uint64_t* d_array;
    CHECK_CUDA(cudaMalloc(&d_array, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_array, h_array.data(), total_bytes, cudaMemcpyHostToDevice));

    // Launch Configuration
    // 256 threads per block as requested
    int threads_per_block = 256;
    // Calculate grid size to cover the array or saturate GPU
    // For large arrays (DRAM sweep), we want enough blocks to occupy the GPU.
    // For pointer chasing, we generally match the array size or cap at a reasonable saturation point.
    // Here we map 1 thread per element start point up to a max to ensure saturation without excessive overhead?
    // User said: "Launch kernel with optimal block size". 
    // We will launch enough blocks to cover the array size, to ensure we touch the whole memory range.
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // Cap blocks to avoid launch timeouts on massive arrays if not needed, 
    // but for DRAM sweep we specifically want to touch the full footprint.
    // So we keep blocks proportional to size.
    
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
