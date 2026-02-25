import subprocess
import time
import threading
import csv
import argparse
import sys
import os
try:
    import pynvml
except ImportError:
    print("Error: 'nvidia-ml-py' is not installed. Please run: pip install nvidia-ml-py")
    sys.exit(1)

# Configuration
TARGET_DURATION_SEC = 3.0  # Run benchmarks for at least 3 seconds
POWER_SAMPLE_INTERVAL = 0.01 # Sample every 10ms

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

def run_power_test(benchmark_type, executable_name, size_mb, stride_bytes, power_monitor):
    print(f"Running {executable_name}: {size_mb} MB...", end="", flush=True)
    
    # 1. Calibrate
    iters = calibrate_iterations(executable_name, size_mb, stride_bytes)
    # print(f" (iters={iters}) ", end="") 
    
    # 2. Start Power Monitor
    power_monitor.start()
    
    # 3. Run Benchmark
    exe_path = f"./{executable_name}"
    if os.name == 'nt': exe_path += ".exe"
    
    cmd = [exe_path, str(size_mb), str(stride_bytes), str(iters)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 4. Stop Monitor
    avg_power, max_power = power_monitor.stop()
    
    # 5. Parse Bandwidth from output
    try:
        # Looking for: "Effective Bandwidth: 123.45 GB/s"
        import re
        match = re.search(r"Effective Bandwidth:\s+([\d\.]+)\s+GB/s", result.stdout)
        bandwidth = float(match.group(1)) if match else 0.0
    except:
        bandwidth = 0.0
        
    print(f" Done. BW: {bandwidth:.2f} GB/s | Power: {avg_power:.2f} W")
    
    return {
        "Benchmark": benchmark_type,
        "Size_MB": size_mb,
        "Bandwidth_GBs": bandwidth,
        "Avg_Power_W": avg_power,
        "Energy_pJ_bit": (avg_power / (bandwidth * 1e9 * 8)) * 1e12 if bandwidth > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="sm_80", help="GPU Arch")
    parser.add_argument("--csv", type=str, default="power_results.csv")
    parser.add_argument("--max_size_mb", type=float, default=1024.0, help="Maximum array size (MB). Set to VRAM capacity - 2GB.")
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
    # L1: 0.03MB (32KB), 0.0625MB (64KB), 0.25MB (256KB) - covering L1 range
    # L2: 4MB, 16MB, 50MB - covering L2 range
    
    all_sizes = [0.03125, 0.0625, 0.25, 4.0, 16.0, 50.0] 
    
    # Add large sizes if allowed
    if args.max_size_mb >= 10240:
        all_sizes.extend([1024.0, 4096.0, 10240.0])
    elif args.max_size_mb >= 4096:
        all_sizes.extend([1024.0, 4096.0])
    elif args.max_size_mb >= 1024:
        all_sizes.append(1024.0)
        
    results = []
    
    print(f"\n>>> Starting Power Sweep (Target: {TARGET_DURATION_SEC}s per run) <<<\n")
    
    for size in all_sizes:
        # Load
        # Stride 128 for Load (matches sweep)
        res_ld = run_power_test("Load", exe_ld, size, 128, monitor)
        results.append(res_ld)
        
        # Store
        # Stride 32 for Store (matches sweep)
        res_st = run_power_test("Store", exe_st, size, 32, monitor)
        results.append(res_st)
        
    # Save CSV
    if results:
        keys = results[0].keys()
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(results)
            
    print(f"\nResults saved to {args.csv}")

if __name__ == "__main__":
    main()
