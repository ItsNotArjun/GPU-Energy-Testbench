# GPU Energy Consumption Analysis - Microbenchmark Replication

This project contains CUDA microbenchmarks designed to replicate the methodology from the paper **"Analyzing GPU Energy Consumption in Data Movement and Storage" (Delestrac et al., 2024)**.

The goal is to generate compiled binaries that measure latency, throughput, and energy consumption across different levels of the GPU memory hierarchy (L1 Cache, L2 Cache, DRAM).

## Project Structure

*   **`ld.cu` (Load Benchmark)**:
    *   Measures Read Latency and Throughput.
    *   Uses **Pointer Chasing** (`ptr = *ptr`) to force serial execution and defeat hardware prefetching.
    *   Uses inline PTX (`ld.global.u64`) to bypass compiler optimizations.
    *   **Usage**: `ld_benchmark <Size MB> <Stride Bytes> <Iterations>`

*   **`st.cu` (Store Benchmark)**:
    *   Measures Write Bandwidth and Energy.
    *   Uses **Strided Linear Access** to saturate write bandwidth.
    *   Uses inline PTX (`st.global.u64`).
    *   **Usage**: `st_benchmark <Size MB> <Stride Bytes> <Iterations>`

*   **`run_sweep.py` (Automation Script)**:
    *   Compiles the CUDA files (`nvcc -O3`).
    *   Runs a sweep of array sizes (16KB to 1GB) to target L1, L2, and DRAM.
    *   Parses and prints the effective bandwidth for each size.

## Prerequisites

1.  **NVIDIA CUDA Toolkit** (Version 11.x or 12.x).
2.  **Microsoft Visual Studio Build Tools** (for the C++ compiler `cl.exe`).
3.  **Python 3.x**.
4.  An NVIDIA GPU (code defaults to Compute Capability 8.0/Ampere, e.g., A100/A10/RTX 30 series).
    *   *Note: If using a different GPU architecture, edit `run_sweep.py` and change `-arch=sm_80` to match your device (e.g., `sm_75` for Turing, `sm_70` for Volta).*

## How to Run

### Windows (Important)
You **must** run these scripts from the **Visual Studio Developer Command Prompt** so that `nvcc` can find the C++ compiler (`cl.exe`).

1.  Open **"x64 Native Tools Command Prompt for VS 2019"** (or 2022) from the Start Menu.
2.  Navigate to this directory:
    ```cmd
    cd "path\to\replication"
    ```
5.  Run the automation script:
    ```cmd
    python run_sweep.py --max_size_mb 1024
    ```

### Running on Different GPUs (Arguments)

The scripts now support command-line arguments to tailor the test to your hardware (VRAM capacity and Architecture).

| Argument | Description | Example (Laptop) | Example (A100) |
| :--- | :--- | :--- | :--- |
| `--arch` | GPU Architecture Code | `sm_86` (RTX 3050) | `sm_80` (A100) |
| `--max_size_mb` | Max Allocation Size. Set to `VRAM - 2GB`. | `2048` (for 4GB card) | `75000` (for 80GB card) |
| `--csv` | Output filename | `laptop.csv` | `a100.csv` |

#### Examples:

**1. On a Laptop (RTX 3050/4060 - 4GB/8GB VRAM):**
```cmd
# Limit to 2GB to prevent crashing/swapping
python run_sweep.py --arch sm_86 --max_size_mb 2048 --csv laptop_results.csv
```

**2. On a Server (NVIDIA A100 - 40GB VRAM):**
```cmd
# Test up to 35GB to stress full DRAM
python run_sweep.py --arch sm_80 --max_size_mb 35000 --csv a100_results.csv
```

**3. Power Measurement:**
To measure Watts and Joules/Bit (requires `pip install nvidia-ml-py`):
```cmd
python measure_power.py --max_size_mb 2048 --csv power_results.csv
```
