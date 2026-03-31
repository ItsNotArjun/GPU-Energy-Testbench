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


def canonical_benchmark(value):
    v = str(value).strip().lower()
    if v in {"ld_benchmark", "load"}:
        return "Load"
    if v in {"st_benchmark", "store"}:
        return "Store"
    return str(value)


def load_csv_checked(results_dir, filename):
    path = results_dir / filename
    if not path.exists():
        print(f"Error: Missing required file: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS[filename] if c not in df.columns]
    if missing_cols:
        print(f"Error: File {path} is missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"Loaded {filename}: shape={df.shape}, columns={list(df.columns)}")
    return df


def prepare_dataframes(results_dir):
    data = {}
    data["sweep32"] = load_csv_checked(results_dir, "sweep_stride32.csv")
    data["sweep8"] = load_csv_checked(results_dir, "sweep_stride8.csv")
    data["power32"] = load_csv_checked(results_dir, "power_stride32.csv")
    data["power8"] = load_csv_checked(results_dir, "power_stride8.csv")
    data["cal32"] = load_csv_checked(results_dir, "calibrate_stride32.csv")

    for key in ["sweep32", "sweep8", "power32", "power8", "cal32"]:
        data[key] = data[key].copy()
        data[key]["Benchmark_Canon"] = data[key]["Benchmark"].map(canonical_benchmark)

    for key in ["sweep32", "sweep8", "power32", "power8", "cal32"]:
        data[key]["Size_MB"] = pd.to_numeric(data[key]["Size_MB"], errors="coerce")
        data[key]["Bandwidth_GBs"] = pd.to_numeric(data[key]["Bandwidth_GBs"], errors="coerce")

    for key in ["power32", "power8"]:
        data[key]["Static_Power_W"] = pd.to_numeric(data[key]["Static_Power_W"], errors="coerce")
        data[key]["Avg_Kernel_Power_W"] = pd.to_numeric(data[key]["Avg_Kernel_Power_W"], errors="coerce")
        data[key]["Dynamic_Power_W"] = pd.to_numeric(data[key]["Dynamic_Power_W"], errors="coerce")
        data[key]["Mean_Dynamic_Energy_J"] = pd.to_numeric(data[key]["Mean_Dynamic_Energy_J"], errors="coerce")
        data[key]["Std_Dynamic_Energy_J"] = pd.to_numeric(data[key]["Std_Dynamic_Energy_J"], errors="coerce")
        data[key]["Dynamic_Energy_pJ_bit"] = pd.to_numeric(data[key]["Dynamic_Energy_pJ_bit"], errors="coerce")

    data["cal32"]["Iterations"] = pd.to_numeric(data["cal32"]["Iterations"], errors="coerce")

    return data


def save_figure(fig, figures_dir, base_name, generated_files):
    for ext in ["png", "pdf"]:
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
        ax.annotate(
            "Store BW collapses >8GB\n(write buffer saturation)",
            xy=(x_c, y_c),
            xycoords="data",
            xytext=(0.53, 0.38),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=COLOR_STORE, lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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

    ax.annotate(
        "At stride=8B, stores are fully coalesced\n"
        "(4 threads share one 32B sector).\n"
        "Load chains are shorter, so store BW dominates.",
        xy=(4.0, sweep8["Bandwidth_GBs"].median()),
        xycoords="data",
        xytext=(0.47, 0.25),
        textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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
    axes[0].annotate(
        "stride=32 gives higher BW in L2/DRAM\n"
        "because longer chains traverse more unique cache lines",
        xy=(64, load32["Bandwidth_GBs"].median()),
        xytext=(0.08, 0.15),
        textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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
    axes[1].annotate(
        "stride=8 gives higher BW\n(coalesced writes, 4 threads/sector)",
        xy=(64, store8["Bandwidth_GBs"].median()),
        xytext=(0.08, 0.18),
        textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    axes[1].legend()

    fig.tight_layout()
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
        ax.annotate(
            "Store power rises sharply\nin DRAM range\n(~90-99W peak)\nvs load staying near idle (~30-52W)",
            xy=(peak_row["Size_MB"], peak_row["Avg_Kernel_Power_W"]),
            xytext=(0.56, 0.60),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=COLOR_STORE, lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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

    ax.text(
        0.03,
        0.03,
        "Note: NVML resolution ~5-10ms.\n"
        "Points only valid where\n"
        "kernel runtime >> NVML sample interval.\n"
        "Large array sizes omitted (dynamic\n"
        "power below noise floor).",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        ha="left",
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
                    xytext=(0.35, 0.15),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=COLOR_STORE, lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
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
            0.97,
            f"Load: {load_mean:.1f} +/- {load_std:.1f}\n"
            f"Store: {store_mean:.1f} +/- {store_std:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        ax.set_xscale("log")
        ax.set_title(title)
        ax.legend(loc="best")

    fig.supylabel("Effective Bandwidth (GB/s)")
    fig.supxlabel("Iterations")
    fig.suptitle("Calibration: Bandwidth Stability vs Iterations - stride=32B")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

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

    ax.annotate(
        "Above 8GB: write-combining buffers saturate.\n"
        "Controller issues read-modify-write cycles.\n"
        "Bandwidth drops from\n"
        "~270 to ~58 GB/s at stride=32\n"
        "from ~1360 to ~480 GB/s at stride=8",
        xy=(12288, st32[st32["Size_MB"] == 12288]["Bandwidth_GBs"].iloc[0] if (st32["Size_MB"] == 12288).any() else st32["Bandwidth_GBs"].iloc[-1]),
        xytext=(0.06, 0.12),
        textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color=COLOR_STORE, lw=1.1),
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


def main():
    parser = argparse.ArgumentParser(description="Generate all Run-02 paper-style figures from CSV files.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing sweep_stride32.csv, sweep_stride8.csv, power_stride32.csv, power_stride8.csv, calibrate_stride32.csv")
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

    print("\nSummary of generated figures:")
    for name in generated_files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
