import subprocess
import re
import sys
import os

import argparse
import datetime
import csv

def compile_benchmarks(arch):
    print(f"Compiling Benchmarks for {arch}...")
    
    # Compile ld.cu
    cmd_ld = [
        "nvcc", "-O3", f"-arch={arch}", 
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
        "nvcc", "-O3", f"-arch={arch}", 
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

def run_sweep(benchmark_name, stride_bytes=32, max_size_mb=1024):
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

    # Additional Points for Big GPUs (>4GB VRAM)
    # These are only added if max_size_mb allows it
    big_points = [4096, 10240, 20480, 40960, 81920]
    
    # Filter points based on max_size_mb
    dram_points.extend([p for p in big_points if p <= max_size_mb])
    
    # Ensure standard points are also filtered just in case user sets very low limit
    dram_points = [p for p in dram_points if p <= max_size_mb]
    
    all_points = [
        (l1_points, "L1 Cache"),
        (l2_points, "L2 Cache"),
        (dram_points, "DRAM")
    ]

    # Iterations count tuning
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
            results.append({
                "Benchmark": benchmark_name,
                "Size_MB": size,
                "Bandwidth_GBs": bw,
                "Target_Level": label,
                "Stride_Bytes": stride_bytes
            })
    
    print("\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run GPU Memory Microbenchmarks w/ CSV Output")
    parser.add_argument("--arch", type=str, default="sm_80", help="GPU Architecture (sm_80 for A100/30-series, sm_70 for V100, sm_75 for T4)")
    parser.add_argument("--csv", type=str, default="results.csv", help="Output CSV filename")
    parser.add_argument("--max_size_mb", type=float, default=1024.0, help="Maximum array size in MB to test (Default: 1024). Increase for A100/H100.")
    
    args = parser.parse_args()
    
    compile_benchmarks(args.arch)
    
    # Prepare CSV
    all_results = []
    
    # Run Load Benchmark Sweep
    print(">>> Starting Load Benchmark (ld_benchmark) <<<")
    # For pointer chasing, higher stride often defeats prefetchers better, 
    # but the logic inside ld.cu handles the dependency chain regardless of stride value logic-wise.
    # We stick to paper values.
    load_results = run_sweep("ld_benchmark", stride_bytes=128, max_size_mb=args.max_size_mb)
    all_results.extend(load_results)

    # Run Store Benchmark Sweep
    print(">>> Starting Store Benchmark (st_benchmark) <<<")
    store_results = run_sweep("st_benchmark", stride_bytes=32, max_size_mb=args.max_size_mb)
    all_results.extend(store_results)
    
    # Save to CSV
    keys = all_results[0].keys()
    with open(args.csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_results)
        
    print(f"Completed! Results saved to {args.csv}")

if __name__ == "__main__":
    main()
