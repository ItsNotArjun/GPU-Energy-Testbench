import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create graphs directory if it doesn't exist
os.makedirs('graphs', exist_ok=True)

# Read the CSV files
df_sweep = pd.read_csv('rtx5000_sweep.csv')
df_sweep_24gb = pd.read_csv('rtx5000_sweep_24gb.csv')

# Combine both datasets
df = pd.concat([df_sweep, df_sweep_24gb], ignore_index=True)

# Separate by benchmark type
ld_data = df[df['Benchmark'] == 'ld_benchmark'].sort_values('Size_MB')
st_data = df[df['Benchmark'] == 'st_benchmark'].sort_values('Size_MB')

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Define cache boundaries (in MB)
l1_end = 0.25
l2_start = 1
l2_end = 64
dram_start = 100

# Colors for cache levels
l1_color = '#FFE5CC'  # Light orange
l2_color = '#CCE5FF'  # Light blue
dram_color = '#E5CCFF'  # Light purple

# ============= SUBPLOT 1: Load Benchmark (ld) =============
ax = axes[0]

# Add background shading for cache levels
ax.axvspan(0.01, l1_end, alpha=0.2, color='orange', label='L1 Cache Region')
ax.axvspan(l2_start, l2_end, alpha=0.2, color='blue', label='L2 Cache Region')
ax.axvspan(dram_start, df['Size_MB'].max() * 1.1, alpha=0.2, color='purple', label='DRAM Region')

# Plot data
ax.plot(ld_data['Size_MB'], ld_data['Bandwidth_GBs'], 'o-', linewidth=2, markersize=6, label='ld_benchmark', color='darkred')

# Add vertical lines at boundaries
ax.axvline(l1_end, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'L1/L2 Boundary ({l1_end} MB)')
ax.axvline(l2_start, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(l2_end, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'L2/DRAM Boundary (~{l2_end} MB)')
ax.axvline(dram_start, color='purple', linestyle='--', linewidth=2, alpha=0.7)

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Array Size (MB)', fontsize=12, fontweight='bold')
ax.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
ax.set_title('Load Benchmark (ld_benchmark) - Cache Level Boundaries', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)

# ============= SUBPLOT 2: Store Benchmark (st) =============
ax = axes[1]

# Add background shading for cache levels
ax.axvspan(0.01, l1_end, alpha=0.2, color='orange', label='L1 Cache Region')
ax.axvspan(l2_start, l2_end, alpha=0.2, color='blue', label='L2 Cache Region')
ax.axvspan(dram_start, df['Size_MB'].max() * 1.1, alpha=0.2, color='purple', label='DRAM Region')

# Plot data
ax.plot(st_data['Size_MB'], st_data['Bandwidth_GBs'], 's-', linewidth=2, markersize=6, label='st_benchmark', color='darkgreen')

# Add vertical lines at boundaries
ax.axvline(l1_end, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'L1/L2 Boundary ({l1_end} MB)')
ax.axvline(l2_start, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(l2_end, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'L2/DRAM Boundary (~{l2_end} MB)')
ax.axvline(dram_start, color='purple', linestyle='--', linewidth=2, alpha=0.7)

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Array Size (MB)', fontsize=12, fontweight='bold')
ax.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
ax.set_title('Store Benchmark (st_benchmark) - Cache Level Boundaries', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('graphs/cache_hierarchy_boundaries.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved to: graphs/cache_hierarchy_boundaries.png")
plt.close()

# ============= COMBINED GRAPH =============
fig, ax = plt.subplots(figsize=(14, 8))

# Add background shading
ax.axvspan(0.01, l1_end, alpha=0.15, color='orange')
ax.axvspan(l2_start, l2_end, alpha=0.15, color='blue')
ax.axvspan(dram_start, df['Size_MB'].max() * 1.1, alpha=0.15, color='purple')

# Plot both benchmarks
ax.plot(ld_data['Size_MB'], ld_data['Bandwidth_GBs'], 'o-', linewidth=2.5, markersize=7, label='Load (ld_benchmark)', color='darkred')
ax.plot(st_data['Size_MB'], st_data['Bandwidth_GBs'], 's-', linewidth=2.5, markersize=7, label='Store (st_benchmark)', color='darkgreen')

# Add annotations for cache regions
ax.text(0.05, ax.get_ylim()[1] * 0.95, 'L1 Cache\n(0-256 KB)', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
ax.text(10, ax.get_ylim()[1] * 0.95, 'L2 Cache\n(1-64 MB)', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
ax.text(1000, ax.get_ylim()[1] * 0.95, 'DRAM\n(100+ MB)', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

# Add vertical lines at boundaries
ax.axvline(l1_end, color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
ax.axvline(l2_start, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
ax.axvline(l2_end, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
ax.axvline(dram_start, color='purple', linestyle='--', linewidth=2.5, alpha=0.8)

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Array Size (MB)', fontsize=13, fontweight='bold')
ax.set_ylabel('Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax.set_title('RTX 5000 Memory Hierarchy - Load vs Store Performance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax.legend(fontsize=12, loc='best')

plt.tight_layout()
plt.savefig('graphs/combined_cache_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved to: graphs/combined_cache_analysis.png")
plt.close()

# Print summary
print("\n" + "="*60)
print("CACHE HIERARCHY SUMMARY (RTX 5000)")
print("="*60)
print(f"L1 Cache Region:  0.01 MB - {l1_end} MB (256 KB)")
print(f"L2 Cache Region:  {l2_start} MB - {l2_end} MB")
print(f"DRAM Region:      {dram_start} MB and above")
print("="*60)
