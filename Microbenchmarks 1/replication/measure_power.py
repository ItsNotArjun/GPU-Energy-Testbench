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
    print(f"Compiling {name}.cu...")
    cmd = ["nvcc", "-O3", f"-arch={arch}", f"{name}.cu", "-o", name]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling {name}.cu:")
        print(res.stderr)
        sys.exit(1)

def calibrate_iterations(executable, size_mb, stride_bytes):
    """Run a quick test to estimate iterations needed for TARGET_DURATION_SEC"""
    test_iters = 1000
    
    exe_path = f"./{executable}"
    if os.name == 'nt': exe_path += ".exe"
    
    # Run once to warm up driver
    subprocess.run([exe_path, str(size_mb), str(stride_bytes), "1"], capture_output=True)
    
    start = time.time()
    subprocess.run([exe_path, str(size_mb), str(stride_bytes), str(test_iters)], capture_output=True)
    duration = time.time() - start
    
    if duration == 0: duration = 0.001
    
    # Calculate needed iterations
    # If 1000 iters took 0.1s, we need 30000 iters for 3s
    needed_iters = int((TARGET_DURATION_SEC / duration) * test_iters)
    return max(needed_iters, 1000)

def run_power_test(benchmark_name, size_mb, stride_bytes, power_monitor):
    print(f"Running {benchmark_name}: {size_mb} MB...", end="", flush=True)
    
    # 1. Calibrate
    iters = calibrate_iterations(benchmark_name, size_mb, stride_bytes)
    
    # 2. Start Power Monitor
    power_monitor.start()
    
    # 3. Run Benchmark
    exe_path = f"./{benchmark_name}"
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
        "Benchmark": benchmark_name,
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
    
    monitor = PowerMonitor()
    
    compile_benchmark("ld", args.arch)
    compile_benchmark("st", args.arch)
    
    # Sweep Points
    # Standard Sweep (Safe for small GPUs like RTX 3050/4060)
    sizes = [0.015625, 0.0625, 0.25, 1.0, 4.0, 16.0, 50.0, 500.0]
    
    # Additional Points for Big GPUs
    big_points = [1024.0, 4096.0, 10240.0, 20480.0, 40960.0]
    
    # Add big points if allowed by limit
    sizes.extend([p for p in big_points if p <= args.max_size_mb])
    
    # Ensure standard points are filtered too
    sizes = [s for s in sizes if s <= args.max_size_mb]
    
    results = []
    
    print("\n>>> Starting Power Sweep <<<\n")
    
    for size in sizes:
        # Load
        res_ld = run_power_test("ld", size, 128, monitor)
        res_ld["Type"] = "Load"
        results.append(res_ld)
        
        # Store
        res_st = run_power_test("st", size, 32, monitor)
        res_st["Type"] = "Store"
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
