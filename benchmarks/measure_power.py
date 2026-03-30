import subprocess
import time
import threading
import csv
import argparse
import sys
import os
import re
try:
    import pynvml
except ImportError:
    print("Error: 'nvidia-ml-py' is not installed. Please run: pip install nvidia-ml-py")
    sys.exit(1)

# Configuration
TARGET_DURATION_SEC = 3.0  # Run benchmarks for at least 3 seconds
POWER_SAMPLE_INTERVAL = 0.01 # Sample every 10ms
IDLE_WINDOW_SEC = 0.4   # seconds of GPU idle time sampled before each kernel for static power baseline

class PowerMonitor:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.stop_event = threading.Event()
        self.power_readings = []
        self.timestamps = []
        self.handle = None
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.device_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"Monitoring GPU: {self.device_name}")
        except pynvml.NVMLError as err:
            print(f"NVML Init Failed: {err}")
            sys.exit(1)

    def start(self):
        self.stop_event.clear()
        self.power_readings = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self._calculate_stats()

    def _monitor_loop(self):
        start_time = time.time()
        while not self.stop_event.is_set():
            try:
                # Returns power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_readings.append(power_mw / 1000.0) # Convert to Watts
                self.timestamps.append(time.time() - start_time)
            except pynvml.NVMLError:
                pass
            time.sleep(POWER_SAMPLE_INTERVAL)

    def _calculate_stats(self):
        if not self.power_readings:
            return 0.0, 0.0
        
        # Simple average
        avg_power = sum(self.power_readings) / len(self.power_readings)
        max_power = max(self.power_readings)
        
        # Advanced: Filter out "ramp up" and "ramp down" tails? 
        # For now, we assume the benchmark runs long enough that the average is dominated by the active phase.
        
        return avg_power, max_power

def compile_benchmark(name, arch="sm_80"):
    # We compile to "ld_benchmark" to match run_sweep.py convention
    exe_name = f"{name}_benchmark"
    print(f"Compiling {name}.cu into {exe_name}...")
    
    cmd = ["nvcc", "-O3", f"-arch={arch}", f"{name}.cu", "-o", exe_name]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling {name}.cu:")
        print(res.stderr)
        sys.exit(1)
    return exe_name

def calibrate_iterations(executable, size_mb, stride_bytes):
    """
    Run a quick test to estimate iterations needed for TARGET_DURATION_SEC.
    We need the GPU to be busy for ~3-5 seconds to get a stable power reading.
    """
    # Start with a small number of iterations to gauge speed
    # unique to each benchmark type? 
    # ld is slow (random access), st is fast (linear)
    
    if "ld" in executable:
        test_iters = 10  # Random access is very slow
    else:
        test_iters = 1000 # Linear access is fast
    
    exe_path = f"./{executable}"
    if os.name == 'nt': exe_path += ".exe"
    
    # Run once to warm up driver (ignore time)
    subprocess.run([exe_path, str(size_mb), str(stride_bytes), "1"], capture_output=True)
    
    # Run calibration pass
    start = time.time()
    # If test_iters is too small, execution might be dominated by overhead. 
    # We try to run at least a measurable amount.
    subprocess.run([exe_path, str(size_mb), str(stride_bytes), str(test_iters)], capture_output=True)
    duration = time.time() - start
    
    if duration < 0.001: duration = 0.001
    
    # Calculate needed iterations
    # We want TARGET_DURATION_SEC (e.g. 5.0 seconds)
    scale_factor = TARGET_DURATION_SEC / duration
    needed_iters = int(test_iters * scale_factor)
    
    # Safety clamp: Don't run 0 iterations
    return max(needed_iters, 1)

def run_power_test(benchmark_type, executable_name, size_mb, stride_bytes,
                   power_monitor, n_runs=3):
    """
    Measure dynamic energy for one (benchmark, size) combination.

    Steps per run:
      1. Idle window: sample NVML for IDLE_WINDOW_SEC to get static power baseline.
      2. Kernel run: start NVML, launch kernel, stop NVML.
      3. Compute dynamic_power = avg_kernel_power - static_power.
      4. Parse kernel_time_s from the binary's stdout ("Time: X ms").
      5. dynamic_energy_J = dynamic_power * kernel_time_s.
    Repeat n_runs times and report mean, std, min, max of dynamic_energy_J.
    """
    import re
    import statistics

    print(f"  [{benchmark_type}] {size_mb} MB | stride={stride_bytes}B | {n_runs} run(s)...")

    exe_path = f"./{executable_name}"
    if os.name == 'nt':
        exe_path += ".exe"

    # Calibrate iterations once (outside the run loop)
    iters = calibrate_iterations(executable_name, size_mb, stride_bytes)

    dynamic_energies_J = []

    for run_idx in range(n_runs):
        # --- Step 1: Measure static (idle) power ---
        power_monitor.start()
        time.sleep(IDLE_WINDOW_SEC)
        static_avg_w, _ = power_monitor.stop()

        # --- Step 2: Run kernel and measure active power ---
        power_monitor.start()
        cmd = [exe_path, str(size_mb), str(stride_bytes), str(iters)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        kernel_avg_w, _ = power_monitor.stop()

        # --- Step 3: Parse kernel execution time from stdout ---
        time_match = re.search(r"Time:\s+([\d\.]+)\s+ms", result.stdout)
        kernel_time_s = (float(time_match.group(1)) / 1000.0) if time_match else 0.0

        # --- Step 4: Parse bandwidth ---
        bw_match = re.search(r"Effective Bandwidth:\s+([\d\.]+)\s+GB/s", result.stdout)
        bandwidth = float(bw_match.group(1)) if bw_match else 0.0

        # --- Step 5: Compute dynamic energy ---
        dynamic_power_w = max(0.0, kernel_avg_w - static_avg_w)
        dynamic_energy_j = dynamic_power_w * kernel_time_s
        dynamic_energies_J.append(dynamic_energy_j)

        print(f"    run {run_idx+1}: static={static_avg_w:.2f}W  kernel={kernel_avg_w:.2f}W  "
              f"dynamic={dynamic_power_w:.2f}W  t={kernel_time_s:.3f}s  "
              f"dyn_E={dynamic_energy_j*1e3:.3f}mJ  BW={bandwidth:.1f}GB/s")

    # Aggregate across runs
    mean_dyn_e = statistics.mean(dynamic_energies_J)
    std_dyn_e  = statistics.stdev(dynamic_energies_J) if n_runs > 1 else 0.0
    min_dyn_e  = min(dynamic_energies_J)
    max_dyn_e  = max(dynamic_energies_J)

    # Recompute bandwidth and static power from the last run for reporting
    # (bandwidth is stable across runs; static power varies slightly)
    # Compute energy per bit using DYNAMIC energy only:
    # bits_transferred = bandwidth (GB/s) * kernel_time_s * 1e9 * 8
    bits_transferred = bandwidth * kernel_time_s * 1e9 * 8 if (bandwidth > 0 and kernel_time_s > 0) else 1.0
    dynamic_epj_bit  = (mean_dyn_e / bits_transferred) * 1e12 if bits_transferred > 0 else 0.0

    print(f"    => mean_dyn_E={mean_dyn_e*1e3:.3f}mJ  std={std_dyn_e*1e3:.3f}mJ  "
          f"dyn_pJ/bit={dynamic_epj_bit:.2f}")

    return {
        "Benchmark":           benchmark_type,
        "Size_MB":             size_mb,
        "Stride_Bytes":        stride_bytes,
        "Bandwidth_GBs":       bandwidth,
        "Static_Power_W":      static_avg_w,
        "Avg_Kernel_Power_W":  kernel_avg_w,
        "Dynamic_Power_W":     dynamic_power_w,
        "Mean_Dynamic_Energy_J": mean_dyn_e,
        "Std_Dynamic_Energy_J":  std_dyn_e,
        "Min_Dynamic_Energy_J":  min_dyn_e,
        "Max_Dynamic_Energy_J":  max_dyn_e,
        "Dynamic_Energy_pJ_bit": dynamic_epj_bit,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="sm_80", help="GPU Arch")
    parser.add_argument("--csv", type=str, default="power_results.csv")
    parser.add_argument("--max_size_mb", type=float, default=1024.0, help="Maximum array size (MB). Set to VRAM capacity - 2GB.")
    parser.add_argument(
        "--stride", type=int, default=32,
        help="Stride in bytes for BOTH benchmarks. Must match what run_sweep.py uses. Default: 32."
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of repeated measurements per (size, benchmark) point. Results are averaged. Default: 3."
    )
    args = parser.parse_args()
    
    # Initialize NVML
    try:
        import pynvml
        pynvml.nvmlInit()
    except ImportError:
        print("Error: 'nvidia-ml-py' not installed. pip install nvidia-ml-py")
        sys.exit(1)
    except Exception as e:
        print(f"NVML Init failed: {e}")
        sys.exit(1)

    monitor = PowerMonitor(0) # Monitor GPU 0
    
    exe_ld = compile_benchmark("ld", args.arch)
    exe_st = compile_benchmark("st", args.arch)
    
    # Sweep Points (Power Measurement takes time, so we pick key points)
    # L1 (Small), L2 (Medium), DRAM (Large)
    # 0.25 (L1), 4/16/50 (L2), 1GB, 4GB, 8GB, 12GB, 16GB, 24GB
    
    all_sizes = [0.03125, 0.0625, 0.25, 4.0, 16.0, 50.0] 
    
    # Add large sizes up to max
    large_candidates = [1024.0, 4096.0, 8192.0, 12288.0, 16384.0, 24576.0]
    
    for s in large_candidates:
        if s <= args.max_size_mb:
            all_sizes.append(s)

    results = []
    
    print(f"\n>>> Starting Power Sweep (Target: {TARGET_DURATION_SEC}s per run) <<<\n")
    
    for size in all_sizes:
        # Load
        res_ld = run_power_test("Load",  exe_ld, size, args.stride, monitor, n_runs=args.runs)
        results.append(res_ld)
        
        # Store
        res_st = run_power_test("Store", exe_st, size, args.stride, monitor, n_runs=args.runs)
        results.append(res_st)
        
    # Save CSV
    
    if results:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        keys = results[0].keys()
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(results)
            
    print(f"\nResults saved to {args.csv}")

if __name__ == "__main__":
    main()
