import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
})

COLOR_LOAD = "#4878CF"
COLOR_STORE = "#D65F5F"
COLOR_L1 = "#4878CF"
COLOR_L2 = "#F89939"
COLOR_DRAM = "#4DAF4A"
COLOR_ANNOT_NEUTRAL = "#666666"

L1_BOUND_MB = 0.5
DRAM_BOUND_MB = 70.0


REQUIRED_COLUMNS = {
    "sweep_stride32.csv": ["Benchmark", "Size_MB", "Bandwidth_GBs", "Target_Level", "Stride_Bytes"],
    "sweep_stride8.csv": ["Benchmark", "Size_MB", "Bandwidth_GBs", "Target_Level", "Stride_Bytes"],
    "power_stride32.csv": [
        "Benchmark",
        "Size_MB",
        "Stride_Bytes",
        "Bandwidth_GBs",
        "Static_Power_W",
        "Avg_Kernel_Power_W",
        "Dynamic_Power_W",
        "Mean_Dynamic_Energy_J",
        "Std_Dynamic_Energy_J",
        "Min_Dynamic_Energy_J",
        "Max_Dynamic_Energy_J",
        "Dynamic_Energy_pJ_bit",
    ],
    "power_stride8.csv": [
        "Benchmark",
        "Size_MB",
        "Stride_Bytes",
        "Bandwidth_GBs",
        "Static_Power_W",
        "Avg_Kernel_Power_W",
        "Dynamic_Power_W",
        "Mean_Dynamic_Energy_J",
        "Std_Dynamic_Energy_J",
        "Min_Dynamic_Energy_J",
        "Max_Dynamic_Energy_J",
        "Dynamic_Energy_pJ_bit",
    ],
    "calibrate_stride32.csv": ["Benchmark", "Level", "Size_MB", "Iterations", "Stride_Bytes", "Bandwidth_GBs"],
}

REQUIRED_SWEEP_COLUMNS = ["Benchmark", "Size_MB", "Bandwidth_GBs", "Target_Level", "Stride_Bytes"]
REQUIRED_POWER_COLUMNS = [
    "Benchmark",
    "Size_MB",
    "Stride_Bytes",
    "Bandwidth_GBs",
    "Static_Power_W",
    "Avg_Kernel_Power_W",
    "Dynamic_Power_W",
    "Mean_Dynamic_Energy_J",
    "Std_Dynamic_Energy_J",
    "Min_Dynamic_Energy_J",
    "Max_Dynamic_Energy_J",
    "Dynamic_Energy_pJ_bit",
]
REQUIRED_CAL_COLUMNS = ["Benchmark", "Level", "Size_MB", "Iterations", "Stride_Bytes", "Bandwidth_GBs"]


def canonical_benchmark(value):
    v = str(value).strip().lower()
    if v in {"ld_benchmark", "load"}:
        return "Load"
    if v in {"st_benchmark", "store"}:
        return "Store"
    return str(value)


def load_csv_checked(results_dir, filename, required_columns=None):
    path = results_dir / filename
    if not path.exists():
        print(f"Error: Missing required file: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    expected = required_columns
    if expected is None:
        expected = REQUIRED_COLUMNS.get(filename, [])
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        print(f"Error: File {path} is missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"Loaded {filename}: shape={df.shape}, columns={list(df.columns)}")
    return df


def extract_stride_from_filename(filename, prefix):
    stem = Path(filename).stem
    key = f"{prefix}_stride"
    if not stem.startswith(key):
        return None
    suffix = stem[len(key):]
    return int(suffix) if suffix.isdigit() else None


def load_stride_family(results_dir, prefix, required_columns):
    frames = {}
    for csv_path in sorted(results_dir.glob(f"{prefix}_stride*.csv")):
        stride = extract_stride_from_filename(csv_path.name, prefix)
        if stride is None:
            continue
        frames[stride] = load_csv_checked(results_dir, csv_path.name, required_columns)
    if not frames:
        print(f"Error: No {prefix}_stride*.csv files were found in {results_dir}")
        sys.exit(1)
    print(f"Discovered {prefix} strides: {sorted(frames.keys())}")
    return frames


def prepare_dataframes(results_dir):
    data = {}

    sweep_by_stride = load_stride_family(results_dir, "sweep", REQUIRED_SWEEP_COLUMNS)
    power_by_stride = load_stride_family(results_dir, "power", REQUIRED_POWER_COLUMNS)
    cal32 = load_csv_checked(results_dir, "calibrate_stride32.csv", REQUIRED_CAL_COLUMNS)

    if 8 not in sweep_by_stride or 32 not in sweep_by_stride:
        print("Error: sweep_stride8.csv and sweep_stride32.csv are required for baseline figures.")
        sys.exit(1)
    if 8 not in power_by_stride or 32 not in power_by_stride:
        print("Error: power_stride8.csv and power_stride32.csv are required for baseline figures.")
        sys.exit(1)

    norm_sweep = {}
    for stride, df in sweep_by_stride.items():
        cur = df.copy()
        cur["Benchmark_Canon"] = cur["Benchmark"].map(canonical_benchmark)
        cur["Size_MB"] = pd.to_numeric(cur["Size_MB"], errors="coerce")
        cur["Bandwidth_GBs"] = pd.to_numeric(cur["Bandwidth_GBs"], errors="coerce")
        cur["Stride_Bytes"] = pd.to_numeric(cur["Stride_Bytes"], errors="coerce")
        norm_sweep[stride] = cur

    norm_power = {}
    for stride, df in power_by_stride.items():
        cur = df.copy()
        cur["Benchmark_Canon"] = cur["Benchmark"].map(canonical_benchmark)
        cur["Size_MB"] = pd.to_numeric(cur["Size_MB"], errors="coerce")
        cur["Bandwidth_GBs"] = pd.to_numeric(cur["Bandwidth_GBs"], errors="coerce")
        cur["Stride_Bytes"] = pd.to_numeric(cur["Stride_Bytes"], errors="coerce")
        cur["Static_Power_W"] = pd.to_numeric(cur["Static_Power_W"], errors="coerce")
        cur["Avg_Kernel_Power_W"] = pd.to_numeric(cur["Avg_Kernel_Power_W"], errors="coerce")
        cur["Dynamic_Power_W"] = pd.to_numeric(cur["Dynamic_Power_W"], errors="coerce")
        cur["Mean_Dynamic_Energy_J"] = pd.to_numeric(cur["Mean_Dynamic_Energy_J"], errors="coerce")
        cur["Std_Dynamic_Energy_J"] = pd.to_numeric(cur["Std_Dynamic_Energy_J"], errors="coerce")
        cur["Dynamic_Energy_pJ_bit"] = pd.to_numeric(cur["Dynamic_Energy_pJ_bit"], errors="coerce")
        norm_power[stride] = cur

    cal32 = cal32.copy()
    cal32["Benchmark_Canon"] = cal32["Benchmark"].map(canonical_benchmark)
    cal32["Size_MB"] = pd.to_numeric(cal32["Size_MB"], errors="coerce")
    cal32["Bandwidth_GBs"] = pd.to_numeric(cal32["Bandwidth_GBs"], errors="coerce")
    cal32["Iterations"] = pd.to_numeric(cal32["Iterations"], errors="coerce")

    data["sweep_by_stride"] = norm_sweep
    data["power_by_stride"] = norm_power
    data["sweep8"] = norm_sweep[8]
    data["sweep32"] = norm_sweep[32]
    data["power8"] = norm_power[8]
    data["power32"] = norm_power[32]
    data["cal32"] = cal32

    return data


def save_figure(fig, figures_dir, base_name, generated_files):
    for ext in ["png"]:
        out_path = figures_dir / f"{base_name}.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path.name}")
        generated_files.append(out_path.name)


def _top_y_for_labels(ax):
    ymin, ymax = ax.get_ylim()
    if ax.get_yscale() == "log":
        return 10 ** (np.log10(ymin) + 0.94 * (np.log10(ymax) - np.log10(ymin)))
    return ymin + 0.94 * (ymax - ymin)


def _x_center(x0, x1, xscale):
    if xscale == "log":
        x0 = max(x0, 1e-9)
        x1 = max(x1, x0 * 1.0001)
        return np.sqrt(x0 * x1)
    return 0.5 * (x0 + x1)


def annotate_outside(ax, text, xy, xytext_axes=(1.03, 0.5), ha="left", va="center"):
    ax.annotate(
        text,
        xy=xy,
        xycoords="data",
        xytext=xytext_axes,
        textcoords="axes fraction",
        ha=ha,
        va=va,
        annotation_clip=False,
        arrowprops=dict(arrowstyle="->", color=COLOR_ANNOT_NEUTRAL, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )


def add_cache_shading(ax, add_labels=False):
    x0, x1 = ax.get_xlim()
    xscale = ax.get_xscale()

    left_bound = max(x0, 1e-9) if xscale == "log" else x0

    l1_left = left_bound
    l1_right = min(L1_BOUND_MB, x1)
    if l1_right > l1_left:
        ax.axvspan(l1_left, l1_right, color=COLOR_L1, alpha=0.12, zorder=0)

    l2_left = max(L1_BOUND_MB, x0)
    l2_right = min(DRAM_BOUND_MB, x1)
    if l2_right > l2_left:
        ax.axvspan(l2_left, l2_right, color=COLOR_L2, alpha=0.12, zorder=0)

    dram_left = max(DRAM_BOUND_MB, x0)
    dram_right = x1
    if dram_right > dram_left:
        ax.axvspan(dram_left, dram_right, color=COLOR_DRAM, alpha=0.12, zorder=0)

    if x0 < L1_BOUND_MB < x1:
        ax.axvline(L1_BOUND_MB, color="gray", linestyle="--", linewidth=1.0)
    if x0 < DRAM_BOUND_MB < x1:
        ax.axvline(DRAM_BOUND_MB, color="gray", linestyle="--", linewidth=1.0)

    if add_labels:
        y_text = _top_y_for_labels(ax)
        if l1_right > l1_left:
            ax.text(_x_center(l1_left, l1_right, xscale), y_text, "L1",
                    ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))
        if l2_right > l2_left:
            ax.text(_x_center(l2_left, l2_right, xscale), y_text, "L2",
                    ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))
        if dram_right > dram_left:
            ax.text(_x_center(dram_left, dram_right, xscale), y_text, "DRAM",
                    ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))


def _plot_load_store_size_overlay(ax, df, title, add_region_labels=False):
    load_df = df[df["Benchmark_Canon"] == "Load"].sort_values("Size_MB")
    store_df = df[df["Benchmark_Canon"] == "Store"].sort_values("Size_MB")

    ax.plot(load_df["Size_MB"], load_df["Bandwidth_GBs"], color=COLOR_LOAD,
            linestyle="-", marker="o", linewidth=1.8, label="Load")
    ax.plot(store_df["Size_MB"], store_df["Bandwidth_GBs"], color=COLOR_STORE,
            linestyle="-", marker="s", linewidth=1.8, label="Store")

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")

    if not load_df.empty or not store_df.empty:
        all_x = pd.concat([load_df["Size_MB"], store_df["Size_MB"]]).dropna()
        if not all_x.empty:
            ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)

    add_cache_shading(ax, add_labels=add_region_labels)
    ax.legend()


def plot_fig01(sweep32, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _plot_load_store_size_overlay(
        ax,
        sweep32,
        "Bandwidth vs Array Size - RTX 5000 Ada, stride=32B",
        add_region_labels=True,
    )

    store_df = sweep32[sweep32["Benchmark_Canon"] == "Store"].sort_values("Size_MB")
    collapse_points = store_df[store_df["Size_MB"] >= 12288]
    if not collapse_points.empty:
        x_c = collapse_points.iloc[0]["Size_MB"]
        y_c = collapse_points.iloc[0]["Bandwidth_GBs"]
        annotate_outside(
            ax,
            "Store BW collapses >8GB\n(write buffer saturation)",
            xy=(x_c, y_c),
            xytext_axes=(1.02, 0.30),
        )

    save_figure(fig, figures_dir, "fig01_bandwidth_stride32", generated_files)
    plt.close(fig)


def plot_fig02(sweep8, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _plot_load_store_size_overlay(
        ax,
        sweep8,
        "Bandwidth vs Array Size - RTX 5000 Ada, stride=8B",
        add_region_labels=True,
    )

    store_df = sweep8[sweep8["Benchmark_Canon"] == "Store"].sort_values("Size_MB")
    if not store_df.empty:
        ann_x = store_df.iloc[min(len(store_df) - 1, len(store_df) // 2)]["Size_MB"]
        ann_y = store_df.iloc[min(len(store_df) - 1, len(store_df) // 2)]["Bandwidth_GBs"]
    else:
        ann_x = 1.0
        ann_y = sweep8["Bandwidth_GBs"].median()

    annotate_outside(
        ax,
        "At stride=8B, stores are fully coalesced\n"
        "(4 threads share one 32B sector).\n"
        "Load chains are shorter, so store BW dominates.",
        xy=(ann_x, ann_y),
        xytext_axes=(1.02, 0.68),
    )

    save_figure(fig, figures_dir, "fig02_bandwidth_stride8", generated_files)
    plt.close(fig)


def plot_fig03(sweep8, sweep32, figures_dir, generated_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    load8 = sweep8[sweep8["Benchmark_Canon"] == "Load"].sort_values("Size_MB")
    load32 = sweep32[sweep32["Benchmark_Canon"] == "Load"].sort_values("Size_MB")

    axes[0].plot(load8["Size_MB"], load8["Bandwidth_GBs"], color=COLOR_LOAD,
                 linestyle="--", marker="o", linewidth=1.8, label="stride=8")
    axes[0].plot(load32["Size_MB"], load32["Bandwidth_GBs"], color=COLOR_LOAD,
                 linestyle="-", marker="o", linewidth=1.8, label="stride=32")
    axes[0].set_xscale("log")
    axes[0].set_title("(a) Load benchmark - stride effect")
    axes[0].set_xlabel("Array Size (MB)")
    axes[0].set_ylabel("Effective Bandwidth (GB/s)")
    add_cache_shading(axes[0], add_labels=False)
    if not load32.empty:
        ann_load_x = load32.iloc[min(len(load32) - 1, len(load32) // 2)]["Size_MB"]
        ann_load_y = load32.iloc[min(len(load32) - 1, len(load32) // 2)]["Bandwidth_GBs"]
    else:
        ann_load_x = 1.0
        ann_load_y = 0.0

    annotate_outside(
        axes[0],
        "stride=32 gives higher BW in L2/DRAM\n"
        "because longer chains traverse more unique cache lines",
        xy=(ann_load_x, ann_load_y),
        xytext_axes=(0.02, -0.26),
        ha="left",
        va="top",
    )
    axes[0].legend()

    store8 = sweep8[sweep8["Benchmark_Canon"] == "Store"].sort_values("Size_MB")
    store32 = sweep32[sweep32["Benchmark_Canon"] == "Store"].sort_values("Size_MB")

    axes[1].plot(store8["Size_MB"], store8["Bandwidth_GBs"], color=COLOR_STORE,
                 linestyle="--", marker="s", linewidth=1.8, label="stride=8")
    axes[1].plot(store32["Size_MB"], store32["Bandwidth_GBs"], color=COLOR_STORE,
                 linestyle="-", marker="s", linewidth=1.8, label="stride=32")
    axes[1].set_xscale("log")
    axes[1].set_title("(b) Store benchmark - stride effect")
    axes[1].set_xlabel("Array Size (MB)")
    axes[1].set_ylabel("Effective Bandwidth (GB/s)")
    add_cache_shading(axes[1], add_labels=False)
    if not store8.empty:
        ann_store_x = store8.iloc[min(len(store8) - 1, len(store8) // 2)]["Size_MB"]
        ann_store_y = store8.iloc[min(len(store8) - 1, len(store8) // 2)]["Bandwidth_GBs"]
    else:
        ann_store_x = 1.0
        ann_store_y = 0.0

    annotate_outside(
        axes[1],
        "stride=8 gives higher BW\n(coalesced writes, 4 threads/sector)",
        xy=(ann_store_x, ann_store_y),
        xytext_axes=(0.02, -0.26),
        ha="left",
        va="top",
    )
    axes[1].legend()

    fig.subplots_adjust(bottom=0.30, wspace=0.24)
    save_figure(fig, figures_dir, "fig03_stride_comparison_load", generated_files)
    save_figure(fig, figures_dir, "fig03_stride_comparison_store", generated_files)
    plt.close(fig)


def plot_fig04(power32, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    load_df = power32[power32["Benchmark_Canon"] == "Load"].sort_values("Size_MB")
    store_df = power32[power32["Benchmark_Canon"] == "Store"].sort_values("Size_MB")

    ax.plot(load_df["Size_MB"], load_df["Avg_Kernel_Power_W"], color=COLOR_LOAD,
            marker="o", linewidth=1.8, label="Load")
    ax.plot(store_df["Size_MB"], store_df["Avg_Kernel_Power_W"], color=COLOR_STORE,
            marker="s", linewidth=1.8, label="Store")

    static_baseline = power32["Static_Power_W"].mean()
    ax.axhline(static_baseline, color="gray", linestyle="--", linewidth=1.3,
               label="GPU idle baseline")

    ax.set_xscale("log")
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Average Power (W)")
    ax.set_title("Avg Kernel Power vs Array Size - RTX 5000 Ada, stride=32B")

    x_all = pd.concat([load_df["Size_MB"], store_df["Size_MB"]]).dropna()
    if not x_all.empty:
        ax.set_xlim(x_all.min() * 0.85, x_all.max() * 1.15)

    add_cache_shading(ax, add_labels=False)

    if not store_df.empty:
        peak_row = store_df.loc[store_df["Avg_Kernel_Power_W"].idxmax()]
        annotate_outside(
            ax,
            "Store power rises sharply\nin DRAM range\n(~90-99W peak)\nvs load staying near idle (~30-52W)",
            xy=(peak_row["Size_MB"], peak_row["Avg_Kernel_Power_W"]),
            xytext_axes=(1.02, 0.70),
        )

    ax.legend(loc="best")
    save_figure(fig, figures_dir, "fig04_power_vs_size_stride32", generated_files)
    plt.close(fig)


def plot_fig05(power32, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    valid = power32[power32["Dynamic_Power_W"] > 0].copy()
    valid["Kernel_Time_S"] = np.where(
        valid["Dynamic_Power_W"] > 0,
        valid["Mean_Dynamic_Energy_J"] / valid["Dynamic_Power_W"],
        np.nan,
    )
    valid["Std_Power_W"] = np.where(
        (valid["Std_Dynamic_Energy_J"] > 0) & (valid["Kernel_Time_S"] > 0),
        valid["Std_Dynamic_Energy_J"] / valid["Kernel_Time_S"],
        np.nan,
    )

    for bench, color, marker in [("Load", COLOR_LOAD, "o"), ("Store", COLOR_STORE, "s")]:
        sub = valid[valid["Benchmark_Canon"] == bench].sort_values("Size_MB")
        if sub.empty:
            continue
        ax.plot(sub["Size_MB"], sub["Dynamic_Power_W"], color=color, marker=marker,
                linewidth=1.8, label=bench)

        err_mask = sub["Std_Power_W"].notna() & (sub["Std_Power_W"] > 0)
        if err_mask.any():
            ax.errorbar(
                sub.loc[err_mask, "Size_MB"],
                sub.loc[err_mask, "Dynamic_Power_W"],
                yerr=sub.loc[err_mask, "Std_Power_W"],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                alpha=0.9,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Dynamic Power (W)")
    ax.set_title("Dynamic Power vs Array Size (valid measurements only) - stride=32B")

    if not valid.empty:
        ax.set_xlim(valid["Size_MB"].min() * 0.85, valid["Size_MB"].max() * 1.15)

    add_cache_shading(ax, add_labels=False)

    if not valid.empty:
        note_anchor = valid.sort_values("Size_MB").iloc[0]
        annotate_outside(
            ax,
            "Note: NVML resolution ~5-10ms.\n"
            "Points only valid where\n"
            "kernel runtime >> NVML sample interval.\n"
            "Large array sizes omitted (dynamic\n"
            "power below noise floor).",
            xy=(note_anchor["Size_MB"], note_anchor["Dynamic_Power_W"]),
            xytext_axes=(1.02, 0.18),
            va="bottom",
        )

    ax.legend(loc="best")
    save_figure(fig, figures_dir, "fig05_dynamic_power_stride32", generated_files)
    plt.close(fig)


def plot_fig06(power32, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    valid = power32[power32["Dynamic_Energy_pJ_bit"] > 0].copy()

    for bench, color, marker in [("Load", COLOR_LOAD, "o"), ("Store", COLOR_STORE, "s")]:
        sub = valid[valid["Benchmark_Canon"] == bench].sort_values("Size_MB")
        if sub.empty:
            continue
        ax.plot(sub["Size_MB"], sub["Dynamic_Energy_pJ_bit"], color=color,
                marker=marker, linewidth=1.8, label=bench)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Dynamic Energy (pJ/bit)")
    ax.set_title("Dynamic Energy per Bit vs Array Size - stride=32B")

    if not valid.empty:
        ax.set_xlim(valid["Size_MB"].min() * 0.85, valid["Size_MB"].max() * 1.15)

    add_cache_shading(ax, add_labels=False)

    ref_lines = [
        (0.1, "L1 cache reference"),
        (1.0, "L2 cache reference"),
        (10.0, "DRAM reference (A100 paper value)"),
    ]
    for y_ref, label in ref_lines:
        ax.axhline(y_ref, color="gray", linestyle=":", linewidth=1.0)
        ax.text(0.99, y_ref, label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color="gray", fontsize=9)

    ax.legend(loc="best")
    save_figure(fig, figures_dir, "fig06_dynamic_epjbit_stride32", generated_files)
    plt.close(fig)


def plot_fig07(cal32, figures_dir, generated_files):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    level_order = ["L1 Cache", "L2 Cache", "DRAM"]
    titles = ["(a) L1 Cache", "(b) L2 Cache", "(c) DRAM"]

    for idx, (level, title) in enumerate(zip(level_order, titles)):
        ax = axes[idx]
        level_df = cal32[cal32["Level"] == level].copy()

        load_df = level_df[level_df["Benchmark_Canon"] == "Load"].sort_values("Iterations")
        store_df = level_df[level_df["Benchmark_Canon"] == "Store"].sort_values("Iterations")

        if not load_df.empty:
            ax.plot(load_df["Iterations"], load_df["Bandwidth_GBs"], color=COLOR_LOAD,
                    marker="o", linestyle="-", linewidth=1.8, label="Load")
        if not store_df.empty:
            ax.plot(store_df["Iterations"], store_df["Bandwidth_GBs"], color=COLOR_STORE,
                    marker="s", linestyle="-", linewidth=1.8, label="Store")

        if level == "DRAM":
            outlier = store_df[store_df["Iterations"] == 1]
            if not outlier.empty:
                ox = outlier.iloc[0]["Iterations"]
                oy = outlier.iloc[0]["Bandwidth_GBs"]
                ax.scatter([ox], [oy], color=COLOR_STORE, marker="X", s=110, zorder=5)
                ax.annotate(
                    "outlier (warmup artifact)",
                    xy=(ox, oy),
                    xytext=(1.03, 0.22),
                    textcoords="axes fraction",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="->", color=COLOR_ANNOT_NEUTRAL, lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                )

        load_mean = load_df["Bandwidth_GBs"].mean()
        load_std = load_df["Bandwidth_GBs"].std(ddof=1)

        store_for_stats = store_df
        if level == "DRAM":
            store_for_stats = store_df[store_df["Iterations"] != 1]
        store_mean = store_for_stats["Bandwidth_GBs"].mean()
        store_std = store_for_stats["Bandwidth_GBs"].std(ddof=1)

        load_std = 0.0 if np.isnan(load_std) else load_std
        store_std = 0.0 if np.isnan(store_std) else store_std

        ax.text(
            0.03,
            -0.26,
            f"Load: {load_mean:.1f} +/- {load_std:.1f}\n"
            f"Store: {store_mean:.1f} +/- {store_std:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        ax.set_xscale("log")
        ax.set_title(title)
        ax.legend(loc="best")

    fig.supylabel("Effective Bandwidth (GB/s)")
    fig.supxlabel("Iterations")
    fig.suptitle("Calibration: Bandwidth Stability vs Iterations - stride=32B")
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])

    save_figure(fig, figures_dir, "fig07_calibration_bw_vs_iters", generated_files)
    plt.close(fig)


def level_from_size(size_mb):
    if size_mb < L1_BOUND_MB:
        return "L1"
    if size_mb <= DRAM_BOUND_MB:
        return "L2"
    return "DRAM"


def plot_fig08(sweep32, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    df = sweep32.copy()
    df["Level"] = df["Size_MB"].apply(level_from_size)

    grouped = (
        df.groupby(["Level", "Benchmark_Canon"], as_index=False)["Bandwidth_GBs"]
        .mean()
    )

    levels = ["L1", "L2", "DRAM"]
    x = np.arange(len(levels))
    width = 0.36

    load_vals = []
    store_vals = []
    for lvl in levels:
        load_row = grouped[(grouped["Level"] == lvl) & (grouped["Benchmark_Canon"] == "Load")]
        store_row = grouped[(grouped["Level"] == lvl) & (grouped["Benchmark_Canon"] == "Store")]
        load_vals.append(float(load_row["Bandwidth_GBs"].iloc[0]) if not load_row.empty else np.nan)
        store_vals.append(float(store_row["Bandwidth_GBs"].iloc[0]) if not store_row.empty else np.nan)

    bars_load = ax.bar(x - width / 2, load_vals, width, color=COLOR_LOAD, label="Load")
    bars_store = ax.bar(x + width / 2, store_vals, width, color=COLOR_STORE, label="Store")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Cache Level")
    ax.set_ylabel("Mean Bandwidth (GB/s)")
    ax.set_title("Mean Bandwidth per Cache Level - Load vs Store, stride=32B")
    ax.legend(loc="best")

    for bar in list(bars_load) + list(bars_store):
        height = bar.get_height()
        if np.isnan(height) or height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.06,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for i, (lv, sv) in enumerate(zip(load_vals, store_vals)):
        if np.isnan(lv) or np.isnan(sv) or lv <= 0:
            continue
        ratio = sv / lv
        y_pair = max(lv, sv) * 1.22
        ax.text(x[i], y_pair, f"ST/LD = {ratio:.1f}x", ha="center", va="bottom", fontsize=9)

    save_figure(fig, figures_dir, "fig08_level_comparison_bars", generated_files)
    plt.close(fig)


def plot_fig09(sweep32, sweep8, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    st32 = sweep32[(sweep32["Benchmark_Canon"] == "Store") & (sweep32["Size_MB"] > DRAM_BOUND_MB)].sort_values("Size_MB")
    st8 = sweep8[(sweep8["Benchmark_Canon"] == "Store") & (sweep8["Size_MB"] > DRAM_BOUND_MB)].sort_values("Size_MB")

    ax.plot(st32["Size_MB"], st32["Bandwidth_GBs"], color=COLOR_STORE,
            linestyle="-", marker="s", linewidth=1.8, label="Store stride=32")
    ax.plot(st8["Size_MB"], st8["Bandwidth_GBs"], color=COLOR_STORE,
            linestyle="--", marker="o", linewidth=1.8, label="Store stride=8")

    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("Store Bandwidth Collapse at Large DRAM Sizes - RTX 5000 Ada")

    if not st32.empty or not st8.empty:
        all_x = pd.concat([st32["Size_MB"], st8["Size_MB"]]).dropna()
        if not all_x.empty:
            ax.set_xlim(all_x.min() * 0.95, all_x.max() * 1.03)

    add_cache_shading(ax, add_labels=False)

    ax.axvline(8192, color="gray", linestyle="--", linewidth=1.2)
    ax.text(8192 * 1.005, ax.get_ylim()[1] * 0.95, "BW collapse threshold",
            color="gray", ha="left", va="top", fontsize=9)

    target_y = st32[st32["Size_MB"] == 12288]["Bandwidth_GBs"].iloc[0] if (st32["Size_MB"] == 12288).any() else st32["Bandwidth_GBs"].iloc[-1]
    annotate_outside(
        ax,
        "Above 8GB: write-combining buffers saturate.\n"
        "Controller issues read-modify-write cycles.\n"
        "Bandwidth drops from\n"
        "~270 to ~58 GB/s at stride=32\n"
        "from ~1360 to ~480 GB/s at stride=8",
        xy=(12288, target_y),
        xytext_axes=(1.03, 0.46),
    )

    ax.legend(loc="best")
    save_figure(fig, figures_dir, "fig09_store_dram_collapse", generated_files)
    plt.close(fig)


def _plot_power_panel(ax, df, bench_name, stride_label):
    sub = df[df["Benchmark_Canon"] == bench_name].sort_values("Size_MB")
    color = COLOR_LOAD if bench_name == "Load" else COLOR_STORE

    ax.plot(sub["Size_MB"], sub["Avg_Kernel_Power_W"], color=color,
            linestyle="-", marker="o" if bench_name == "Load" else "s", linewidth=1.8)

    idle = df["Static_Power_W"].mean()
    ax.axhline(idle, color="gray", linestyle="--", linewidth=1.0)
    ax.text(0.98, idle, "idle", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", color="gray", fontsize=9)

    ax.set_xscale("log")
    if not sub.empty:
        ax.set_xlim(sub["Size_MB"].min() * 0.85, sub["Size_MB"].max() * 1.15)

    add_cache_shading(ax, add_labels=False)
    ax.set_ylim(0, 120)
    ax.set_title(f"{bench_name} stride={stride_label}")


def plot_fig10(power32, power8, figures_dir, generated_files):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)

    _plot_power_panel(axes[0, 0], power32, "Load", "32")
    _plot_power_panel(axes[0, 1], power32, "Store", "32")
    _plot_power_panel(axes[1, 0], power8, "Load", "8")
    _plot_power_panel(axes[1, 1], power8, "Store", "8")

    for ax in axes[1, :]:
        ax.set_xlabel("Array Size (MB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Average Power (W)")

    fig.suptitle("Kernel Power vs Array Size - All Stride Configurations")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_figure(fig, figures_dir, "fig10_power_heatmap", generated_files)
    plt.close(fig)


def get_stride_style_map(strides):
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    style_map = {}
    for i, stride in enumerate(sorted(strides)):
        style_map[stride] = (line_styles[i % len(line_styles)], markers[i % len(markers)])
    return style_map


def get_stride_color_map(strides):
    # Fixed high-contrast colors, consistent for a given stride across figures.
    fixed = {
        8: "#1f77b4",
        16: "#ff7f0e",
        32: "#2ca02c",
        64: "#d62728",
        128: "#9467bd",
    }
    fallback = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    out = {}
    fallback_idx = 0
    for stride in sorted(strides):
        if stride in fixed:
            out[stride] = fixed[stride]
        else:
            out[stride] = fallback[fallback_idx % len(fallback)]
            fallback_idx += 1
    return out


def plot_fig11_all_stride_bandwidth(sweep_by_stride, figures_dir, generated_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False)
    style_map = get_stride_style_map(sweep_by_stride.keys())
    color_map = get_stride_color_map(sweep_by_stride.keys())

    for ax, bench, title in [
        (axes[0], "Load", "(a) Load bandwidth across strides"),
        (axes[1], "Store", "(b) Store bandwidth across strides"),
    ]:
        x_vals = []
        for stride in sorted(sweep_by_stride):
            df = sweep_by_stride[stride]
            sub = df[df["Benchmark_Canon"] == bench].sort_values("Size_MB")
            if sub.empty:
                continue
            ls, mk = style_map[stride]
            ax.plot(
                sub["Size_MB"],
                sub["Bandwidth_GBs"],
                color=color_map[stride],
                linestyle=ls,
                marker=mk,
                linewidth=1.7,
                label=f"{stride}B",
            )
            x_vals.append(sub["Size_MB"])

        ax.set_xscale("log")
        if x_vals:
            all_x = pd.concat(x_vals).dropna()
            if not all_x.empty:
                ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)
        add_cache_shading(ax, add_labels=False)
        ax.set_title(title)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Effective Bandwidth (GB/s)")
        ax.legend(title="Stride", ncol=2, fontsize=9)

    fig.suptitle("Bandwidth vs Array Size - all available strides")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, figures_dir, "fig11_all_stride_bandwidth", generated_files)
    plt.close(fig)


def plot_fig12_level_mean_vs_stride(sweep_by_stride, figures_dir, generated_files):
    rows = []
    for stride, df in sweep_by_stride.items():
        cur = df.copy()
        cur["Level"] = cur["Size_MB"].apply(level_from_size)
        grouped = cur.groupby(["Benchmark_Canon", "Level"], as_index=False)["Bandwidth_GBs"].mean()
        for _, row in grouped.iterrows():
            rows.append({
                "Stride": stride,
                "Benchmark_Canon": row["Benchmark_Canon"],
                "Level": row["Level"],
                "Mean_BW": row["Bandwidth_GBs"],
            })

    stats = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    level_colors = {"L1": COLOR_L1, "L2": COLOR_L2, "DRAM": COLOR_DRAM}

    for ax, bench, title in [
        (axes[0], "Load", "(a) Load mean bandwidth by level"),
        (axes[1], "Store", "(b) Store mean bandwidth by level"),
    ]:
        sub_b = stats[stats["Benchmark_Canon"] == bench]
        strides_sorted = sorted(sub_b["Stride"].unique())

        for level in ["L1", "L2", "DRAM"]:
            y_vals = []
            for stride in strides_sorted:
                row = sub_b[(sub_b["Stride"] == stride) & (sub_b["Level"] == level)]
                y_vals.append(float(row["Mean_BW"].iloc[0]) if not row.empty else np.nan)
            ax.plot(
                strides_sorted,
                y_vals,
                color=level_colors[level],
                marker="o",
                linewidth=1.8,
                label=level,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(strides_sorted)
        ax.set_xticklabels([f"{s}B" for s in strides_sorted])
        ax.set_title(title)
        ax.set_xlabel("Stride (bytes)")
        ax.set_ylabel("Mean Bandwidth (GB/s)")
        ax.legend(title="Level")

    fig.suptitle("Cache-level mean bandwidth vs stride")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save_figure(fig, figures_dir, "fig12_level_mean_vs_stride", generated_files)
    plt.close(fig)


def plot_fig13_store_dram_collapse_all_strides(sweep_by_stride, figures_dir, generated_files):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    style_map = get_stride_style_map(sweep_by_stride.keys())

    strides_sorted = sorted(sweep_by_stride.keys())
    shades = np.linspace(0.45, 0.9, len(strides_sorted))
    stride_colors = {s: plt.cm.Reds(shades[i]) for i, s in enumerate(strides_sorted)}

    for stride in strides_sorted:
        df = sweep_by_stride[stride]
        sub = df[(df["Benchmark_Canon"] == "Store") & (df["Size_MB"] > DRAM_BOUND_MB)].sort_values("Size_MB")
        if sub.empty:
            continue
        ls, mk = style_map[stride]
        ax.plot(
            sub["Size_MB"],
            sub["Bandwidth_GBs"],
            color=stride_colors[stride],
            linestyle=ls,
            marker=mk,
            linewidth=1.8,
            label=f"stride={stride}B",
        )

    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Store Bandwidth (GB/s)")
    ax.set_title("Store DRAM collapse comparison across strides")
    ax.axvline(8192, color="gray", linestyle="--", linewidth=1.2)
    ax.text(8192 * 1.004, ax.get_ylim()[1] * 0.96, "8GB", color="gray", ha="left", va="top", fontsize=9)
    add_cache_shading(ax, add_labels=False)

    ref_df = sweep_by_stride[strides_sorted[0]]
    ref = ref_df[(ref_df["Benchmark_Canon"] == "Store") & (ref_df["Size_MB"] > DRAM_BOUND_MB)].sort_values("Size_MB")
    if not ref.empty:
        anchor_x = ref.iloc[min(len(ref) - 1, len(ref) // 2)]["Size_MB"]
        anchor_y = ref.iloc[min(len(ref) - 1, len(ref) // 2)]["Bandwidth_GBs"]
        annotate_outside(
            ax,
            "Compare where each stride begins\n"
            "to collapse in large-DRAM regime\n"
            "(buffer saturation threshold shifts\n"
            "with stride/coalescing).",
            xy=(anchor_x, anchor_y),
            xytext_axes=(1.03, 0.28),
        )

    ax.legend(title="Store stride", ncol=2, fontsize=9)
    save_figure(fig, figures_dir, "fig13_store_dram_collapse_all_strides", generated_files)
    plt.close(fig)


def plot_fig14_power_all_strides(power_by_stride, figures_dir, generated_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=True)
    style_map = get_stride_style_map(power_by_stride.keys())
    color_map = get_stride_color_map(power_by_stride.keys())

    for ax, bench, title in [
        (axes[0], "Load", "(a) Load power across strides"),
        (axes[1], "Store", "(b) Store power across strides"),
    ]:
        x_vals = []
        all_idle_vals = []
        for stride in sorted(power_by_stride):
            df = power_by_stride[stride]
            sub = df[df["Benchmark_Canon"] == bench].sort_values("Size_MB")
            if sub.empty:
                continue
            ls, mk = style_map[stride]
            ax.plot(
                sub["Size_MB"],
                sub["Avg_Kernel_Power_W"],
                color=color_map[stride],
                linestyle=ls,
                marker=mk,
                linewidth=1.7,
                label=f"{stride}B",
            )
            x_vals.append(sub["Size_MB"])
            all_idle_vals.append(df["Static_Power_W"])

        if all_idle_vals:
            idle_mean = pd.concat(all_idle_vals).mean()
            ax.axhline(idle_mean, color="gray", linestyle="--", linewidth=1.0)

        ax.set_xscale("log")
        if x_vals:
            all_x = pd.concat(x_vals).dropna()
            if not all_x.empty:
                ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)
        add_cache_shading(ax, add_labels=False)
        ax.set_title(title)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Average Kernel Power (W)")
        ax.legend(title="Stride", ncol=2, fontsize=9)

    fig.suptitle("Average kernel power vs size - all available strides")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, figures_dir, "fig14_power_all_strides", generated_files)
    plt.close(fig)


def plot_fig15_dynamic_epjbit_all_strides(power_by_stride, figures_dir, generated_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=True)
    style_map = get_stride_style_map(power_by_stride.keys())

    for ax, bench, color, title in [
        (axes[0], "Load", COLOR_LOAD, "(a) Load dynamic energy/bit across strides"),
        (axes[1], "Store", COLOR_STORE, "(b) Store dynamic energy/bit across strides"),
    ]:
        x_vals = []
        for stride in sorted(power_by_stride):
            df = power_by_stride[stride]
            sub = df[
                (df["Benchmark_Canon"] == bench) &
                (df["Dynamic_Energy_pJ_bit"] > 0) &
                (df["Mean_Dynamic_Energy_J"] > 0)
            ].sort_values("Size_MB")
            if sub.empty:
                continue
            ls, mk = style_map[stride]
            ax.plot(
                sub["Size_MB"],
                sub["Dynamic_Energy_pJ_bit"],
                color=color,
                linestyle=ls,
                marker=mk,
                linewidth=1.7,
                label=f"{stride}B",
            )
            x_vals.append(sub["Size_MB"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        if x_vals:
            all_x = pd.concat(x_vals).dropna()
            if not all_x.empty:
                ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)
        add_cache_shading(ax, add_labels=False)
        ax.set_title(title)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Dynamic Energy (pJ/bit)")
        ax.legend(title="Stride", ncol=2, fontsize=9)

    fig.suptitle("Dynamic energy per bit vs size - all available strides (valid points only)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, figures_dir, "fig15_dynamic_epjbit_all_strides", generated_files)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate all Run-02 paper-style figures from CSV files.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing sweep_stride*.csv, power_stride*.csv, and calibrate_stride32.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_dataframes(results_dir)

    generated_files = []

    plot_fig01(data["sweep32"], figures_dir, generated_files)
    plot_fig02(data["sweep8"], figures_dir, generated_files)
    plot_fig03(data["sweep8"], data["sweep32"], figures_dir, generated_files)
    plot_fig04(data["power32"], figures_dir, generated_files)
    plot_fig05(data["power32"], figures_dir, generated_files)
    plot_fig06(data["power32"], figures_dir, generated_files)
    plot_fig07(data["cal32"], figures_dir, generated_files)
    plot_fig08(data["sweep32"], figures_dir, generated_files)
    plot_fig09(data["sweep32"], data["sweep8"], figures_dir, generated_files)
    plot_fig10(data["power32"], data["power8"], figures_dir, generated_files)
    plot_fig11_all_stride_bandwidth(data["sweep_by_stride"], figures_dir, generated_files)
    plot_fig12_level_mean_vs_stride(data["sweep_by_stride"], figures_dir, generated_files)
    plot_fig13_store_dram_collapse_all_strides(data["sweep_by_stride"], figures_dir, generated_files)
    plot_fig14_power_all_strides(data["power_by_stride"], figures_dir, generated_files)
    plot_fig15_dynamic_epjbit_all_strides(data["power_by_stride"], figures_dir, generated_files)

    print("\nSummary of generated figures:")
    for name in generated_files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
