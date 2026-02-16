@echo off
setlocal
echo ==============================================================================
echo Nsight Compute (ncu) Profiling Script for ld.cu & st.cu
echo ==============================================================================
echo Goal: Check if memory accesses are coalesced (1:1 efficiency) or wasteful.
echo.
echo Prerequisites: 
echo   1. 'ncu.exe' must be in PATH. 
echo   2. Run this from x64 Native Tools Command Prompt.
echo   3. Ensure ld_benchmark.exe & st_benchmark.exe are compiled (run run_sweep.py first).
echo.
echo ==============================================================================

:: Check if ncu is available
where ncu >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: 'ncu' command not found.
    echo Please add C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.x\nsight-compute\ to your PATH.
    exit /b 1
)

:: Metric Explanations:
:: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  -> L1/Tex Global Load Sectors (32B chunks requested)
:: l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum  -> L1/Tex Global Store Sectors (32B chunks requested)
:: l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum -> Actual Load Instructions Executed (from your code)
:: l1tex__t_requests_pipe_lsu_mem_global_op_st.sum -> Actual Store Instructions Executed (from your code)
::
:: Ideally: Sectors / Requests should be close to 1.0 (perfect coalescing for 32B types) 
::          or 4.0 if accessing 8B types (fetch 32B just for 8B).
::
:: For 'ld.cu' (Pointer Chasing), we expect Ratio = 4 (fetching 32B sector for 8B pointer).
:: For 'st.cu' (Strided), we expect Ratio = 1 (coalesced 32B writes).

echo.
echo [1/2] Profiling Load Benchmark (Size: 4MB - L2 Target)...
echo Running: ncu --csv --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ld_benchmark.exe 4.0 128 1000
echo ------------------------------------------------------------------------------
ncu --csv --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ld_benchmark.exe 4.0 128 1000 > profile_ld_results.csv
type profile_ld_results.csv

echo.
echo [2/2] Profiling Store Benchmark (Size: 100MB - DRAM Target)...
echo Running: ncu --csv --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum st_benchmark.exe 100.0 32 1000
echo ------------------------------------------------------------------------------
ncu --csv --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum st_benchmark.exe 100.0 32 1000 > profile_st_results.csv
type profile_st_results.csv

echo.
echo ==============================================================================
echo Profiling Complete.
echo Check 'profile_ld_results.csv' and 'profile_st_results.csv'.
echo Use the README guide to interpret the "Sectors per Request" ratio.
echo ==============================================================================
endlocal
