import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create graphs directory if it doesn't exist
os.makedirs('graphs', exist_ok=True)

# Read the power CSV files
df_power = pd.read_csv('rtx5000_power.csv')
df_power_24gb = pd.read_csv('rtx5000_power_24gb.csv')

# Combine both datasets
df = pd.concat([df_power, df_power_24gb], ignore_index=True)

# Separate by benchmark type
load_data = df[df['Benchmark'] == 'Load'].sort_values('Size_MB')
store_data = df[df['Benchmark'] == 'Store'].sort_values('Size_MB')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.delaxes(axes[1, 1])

# ============= SUBPLOT 1: Power vs Array Size =============
ax = axes[0, 0]
ax.plot(load_data['Size_MB'], load_data['Avg_Power_W'], 'o-', linewidth=2.5, markersize=8, label='Load (Stride: 128B)', color='darkred')
ax.plot(store_data['Size_MB'], store_data['Avg_Power_W'], 's-', linewidth=2.5, markersize=8, label='Store (Stride: 32B)', color='darkgreen')

ax.set_xscale('log')
ax.set_xlabel('Array Size (MB)', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
ax.set_title('Power Consumption vs Array Size', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10, loc='best')

# ============= SUBPLOT 2: Energy per Bit vs Array Size =============
ax = axes[0, 1]
ax.plot(load_data['Size_MB'], load_data['Energy_pJ_bit'], 'o-', linewidth=2.5, markersize=8, label='Load (Stride: 128B)', color='darkred')
ax.plot(store_data['Size_MB'], store_data['Energy_pJ_bit'], 's-', linewidth=2.5, markersize=8, label='Store (Stride: 32B)', color='darkgreen')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Array Size (MB)', fontsize=11, fontweight='bold')
ax.set_ylabel('Energy per Bit (pJ/bit)', fontsize=11, fontweight='bold')
ax.set_title('Energy Efficiency vs Array Size', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10, loc='best')

# ============= SUBPLOT 3: Power vs Bandwidth =============
ax = axes[1, 0]
ax.scatter(load_data['Bandwidth_GBs'], load_data['Avg_Power_W'], s=100, alpha=0.7, label='Load (Stride: 128B)', color='darkred', marker='o')
ax.scatter(store_data['Bandwidth_GBs'], store_data['Avg_Power_W'], s=100, alpha=0.7, label='Store (Stride: 32B)', color='darkgreen', marker='s')

# Add size annotations (log scale for better readability)
for idx, row in load_data.iterrows():
    ax.annotate(f"{row['Size_MB']:.0f}MB", (row['Bandwidth_GBs'], row['Avg_Power_W']), 
                fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
for idx, row in store_data.iterrows():
    ax.annotate(f"{row['Size_MB']:.0f}MB", (row['Bandwidth_GBs'], row['Avg_Power_W']), 
                fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

ax.set_xscale('log')
ax.set_xlabel('Bandwidth (GB/s)', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
ax.set_title('Power vs Bandwidth (Stride Impact)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10, loc='best')

# ============= SUBPLOT 4: Bandwidth vs Array Size =============
# ax = axes[1, 1]
# ax.plot(load_data['Size_MB'], load_data['Bandwidth_GBs'], 'o-', linewidth=2.5, markersize=8, label='Load (Stride: 128B)', color='darkred')
# ax.plot(store_data['Size_MB'], store_data['Bandwidth_GBs'], 's-', linewidth=2.5, markersize=8, label='Store (Stride: 32B)', color='darkgreen')

# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('Array Size (MB)', fontsize=11, fontweight='bold')
# ax.set_ylabel('Bandwidth (GB/s)', fontsize=11, fontweight='bold')
# ax.set_title('Bandwidth vs Array Size (Stride Effect)', fontsize=12, fontweight='bold')
# ax.grid(True, alpha=0.3, which='both')
# ax.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('graphs/power_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved to: graphs/power_analysis.png")
plt.close()

# ============= COMPOSITE: Power & Energy per Bit =============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Power consumption
ax1.plot(load_data['Size_MB'], load_data['Avg_Power_W'], 'o-', linewidth=3, markersize=9, label='Load (Stride: 128B)', color='#E63946', markeredgewidth=2)
ax1.plot(store_data['Size_MB'], store_data['Avg_Power_W'], 's-', linewidth=3, markersize=9, label='Store (Stride: 32B)', color='#2A9D8F', markeredgewidth=2)

ax1.fill_between(load_data['Size_MB'], load_data['Avg_Power_W'], alpha=0.2, color='#E63946')
ax1.fill_between(store_data['Size_MB'], store_data['Avg_Power_W'], alpha=0.2, color='#2A9D8F')

ax1.set_xscale('log')
ax1.set_xlabel('Array Size (MB)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Power (W)', fontsize=12, fontweight='bold')
ax1.set_title('Power Consumption by Array Size', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')
ax1.legend(fontsize=11, loc='best')

# Energy per bit
ax2.plot(load_data['Size_MB'], load_data['Energy_pJ_bit'], 'o-', linewidth=3, markersize=9, label='Load (Stride: 128B)', color='#E63946', markeredgewidth=2)
ax2.plot(store_data['Size_MB'], store_data['Energy_pJ_bit'], 's-', linewidth=3, markersize=9, label='Store (Stride: 32B)', color='#2A9D8F', markeredgewidth=2)

ax2.fill_between(load_data['Size_MB'], load_data['Energy_pJ_bit'], alpha=0.2, color='#E63946')
ax2.fill_between(store_data['Size_MB'], store_data['Energy_pJ_bit'], alpha=0.2, color='#2A9D8F')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Array Size (MB)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy per Bit (pJ/bit)', fontsize=12, fontweight='bold')
ax2.set_title('Energy Efficiency by Array Size', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both', linestyle='--')
ax2.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('graphs/power_energy_combined.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved to: graphs/power_energy_combined.png")
plt.close()

# ============= STRIDE COMPARISON TABLE =============
# print("\n" + "="*80)
# print("STRIDE IMPACT ANALYSIS (Load vs Store)")
# print("="*80)
# print(f"{'Size (MB)':<12} {'Load Stride':<12} {'Store Stride':<12} {'Load Power (W)':<15} {'Store Power (W)':<15}")
# print("-"*80)

# Merge load and store data for comparison
# for size in sorted(df['Size_MB'].unique()):
#     load_row = load_data[load_data['Size_MB'] == size]
#     store_row = store_data[store_data['Size_MB'] == size]
    
#     if not load_row.empty and not store_row.empty:
#         print(f"{size:<12.2f} {'128 Bytes':<12} {'32 Bytes':<12} {load_row['Avg_Power_W'].values[0]:<15.2f} {store_row['Avg_Power_W'].values[0]:<15.2f}")

# print("="*80)
# print("\nKEY OBSERVATIONS:")
# print("-"*80)
# print("• Load operations use 128-byte stride (pointer chasing)")
# print("• Store operations use 32-byte stride (coalesced writes)")
# print("• Store operations consistently show higher power at larger sizes (>1GB)")
# print("• This is due to DRAM write pressure being more power-intensive")
# print("="*80)
