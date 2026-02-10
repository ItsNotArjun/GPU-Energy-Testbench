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
    python run_sweep.py
    ```

The script will compile `ld.cu` and `st.cu` into `.exe` files and run the benchmark sweeps, printing the results to the console.
