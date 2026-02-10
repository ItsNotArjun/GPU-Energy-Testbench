import subprocess
import re
import sys
import os

def compile_benchmarks():
    print("Compiling Benchmarks...")
    
    # Compile ld.cu
    cmd_ld = [
        "nvcc", "-O3", "-arch=sm_80", 
        "ld.cu", "-o", "ld_benchmark"
    ]
    print(f"Running: {' '.join(cmd_ld)}")
    res_ld = subprocess.run(cmd_ld, capture_output=True, text=True)
    if res_ld.returncode != 0:
        print("Error compiling ld.cu:")
        print(res_ld.stdout)
        print(res_ld.stderr)
        sys.exit(1)

    # Compile st.cu
    cmd_st = [
        "nvcc", "-O3", "-arch=sm_80", 
        "st.cu", "-o", "st_benchmark"
    ]
    print(f"Running: {' '.join(cmd_st)}")
    res_st = subprocess.run(cmd_st, capture_output=True, text=True)
    if res_st.returncode != 0:
        print("Error compiling st.cu:")
        print(res_st.stdout)
        print(res_st.stderr)
        sys.exit(1)
        
    print("Compilation Successful.\n")

def run_test(executable, size_mb, stride_bytes, iterations):
    # Determine extension for Windows
    exe_path = f"./{executable}"
    if os.name == 'nt':
        exe_path += ".exe"
    
    cmd = [exe_path, str(size_mb), str(stride_bytes), str(iterations)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Regex to extract bandwidth
        # Expected output: "Effective Bandwidth: 123.45 GB/s"
        match = re.search(r"Effective Bandwidth:\s+([\d\.]+)\s+GB/s", result.stdout)
        if match:
            return float(match.group(1))
        else:
            return 0.0
    except subprocess.CalledProcessError as e:
        print(f"Error running {executable}: {e}")
        print(e.stderr)
        return -1.0

def run_sweep(benchmark_name, stride_bytes=32):
    print(f"--- Running Sweep for {benchmark_name} (Stride: {stride_bytes} bytes) ---")
    print(f"{'Size (MB)':<15} {'Bandwidth (GB/s)':<20} {'Target Level'}")
    print("-" * 50)

    # Defined Sweep Ranges based on Requirements
    # L1: 16KB (0.015MB) to 256KB (0.25MB)
    l1_points = [0.015625, 0.03125, 0.0625, 0.125, 0.25] 
    
    # L2: 1MB to 50MB
    l2_points = [1, 2, 4, 8, 16, 24, 32, 40, 50]
    
    # DRAM: 100MB to 1GB
    dram_points = [100, 256, 512, 768, 1024]
    
    all_points = [
        (l1_points, "L1 Cache"),
        (l2_points, "L2 Cache"),
        (dram_points, "DRAM")
    ]

    # Iterations count tuning:
    # Small arrays need more iterations to measure accurately.
    # Large arrays need fewer to save time.
    def get_iters(size_mb):
        if size_mb < 1: return 5000
        if size_mb < 50: return 500
        return 50

    results = []

    for points, label in all_points:
        for size in points:
            iters = get_iters(size)
            bw = run_test(benchmark_name, size, stride_bytes, iters)
            print(f"{size:<15.4f} {bw:<20.4f} {label}")
            results.append((size, bw, label))
    
    print("\n")
    return results

def main():
    compile_benchmarks()
    
    # Run Load Benchmark Sweep
    print(">>> Starting Load Benchmark (ld_benchmark) <<<")
    # For pointer chasing to stress latency/bw, stride implies the random jump distance usually,
    # but here we use linear index + stride.
    # To truly test memory hierarchy latency, stride should be >= Cache Line (32 or 128 bytes).
    run_sweep("ld_benchmark", stride_bytes=128)

    # Run Store Benchmark Sweep
    print(">>> Starting Store Benchmark (st_benchmark) <<<")
    run_sweep("st_benchmark", stride_bytes=32)

if __name__ == "__main__":
    main()
