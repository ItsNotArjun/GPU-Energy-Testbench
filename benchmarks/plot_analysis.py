import argparse
import traceback
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

COL_LOAD = "#4878CF"
COL_STORE = "#D65F5F"
COL_RUN02 = "#888888"
COL_RUN03 = "#2CA02C"
COL_L1_SHADE = "#EEF3FB"
COL_L2_SHADE = "#FFF5E6"
COL_DR_SHADE = "#F0FBF0"
COL_L1_LINE = "#4878CF"
COL_L2_LINE = "#F89939"
COL_DR_LINE = "#4DAF4A"
COL_ANNOT = "#666666"

L1_BOUND_MB = 0.5
DR_BOUND_MB = 70.0


def canonical_benchmark(value):
    v = str(value).strip().lower()
    if v in {"ld_benchmark", "load"}:
        return "Load"
    if v in {"st_benchmark", "store"}:
        return "Store"
    return str(value)


def level_from_size(size_mb):
    if size_mb < L1_BOUND_MB:
        return "L1"
    if size_mb <= DR_BOUND_MB:
        return "L2"
    return "DRAM"


def apply_cache_shading(ax):
    x0, x1 = ax.get_xlim()
    if x0 <= 0:
        x0 = 1e-6

    # Region shading
    l1_left = x0
    l1_right = min(L1_BOUND_MB, x1)
    if l1_right > l1_left:
        ax.axvspan(l1_left, l1_right, color=COL_L1_SHADE, alpha=0.10, zorder=0)

    l2_left = max(L1_BOUND_MB, x0)
    l2_right = min(DR_BOUND_MB, x1)
    if l2_right > l2_left:
        ax.axvspan(l2_left, l2_right, color=COL_L2_SHADE, alpha=0.10, zorder=0)

    dr_left = max(DR_BOUND_MB, x0)
    dr_right = x1
    if dr_right > dr_left:
        ax.axvspan(dr_left, dr_right, color=COL_DR_SHADE, alpha=0.10, zorder=0)

    # Boundaries
    if x0 < L1_BOUND_MB < x1:
        ax.axvline(L1_BOUND_MB, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    if x0 < DR_BOUND_MB < x1:
        ax.axvline(DR_BOUND_MB, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Labels at top of each region
    ymin, ymax = ax.get_ylim()
    if ax.get_yscale() == "log":
        y_text = 10 ** (np.log10(ymin) + 0.95 * (np.log10(ymax) - np.log10(ymin)))
        center = lambda a, b: np.sqrt(max(a, 1e-12) * max(b, 1e-12))
    else:
        y_text = ymin + 0.95 * (ymax - ymin)
        center = lambda a, b: 0.5 * (a + b)

    if l1_right > l1_left:
        ax.text(center(l1_left, l1_right), y_text, "L1", fontsize=9, color=COL_L1_LINE,
                ha="center", va="center")
    if l2_right > l2_left:
        ax.text(center(l2_left, l2_right), y_text, "L2", fontsize=9, color=COL_L2_LINE,
                ha="center", va="center")
    if dr_right > dr_left:
        ax.text(center(dr_left, dr_right), y_text, "DRAM", fontsize=9, color=COL_DR_LINE,
                ha="center", va="center")


def filter_nonzero_energy(df):
    return df[df["Mean_Dynamic_Energy_J"] > 1e-8].copy()


def annotate_outside(ax, text, xy, xytext_axes=(1.02, 0.5), ha="left", va="center"):
    ax.annotate(
        text,
        xy=xy,
        xycoords="data",
        xytext=xytext_axes,
        textcoords="axes fraction",
        ha=ha,
        va=va,
        annotation_clip=False,
        arrowprops=dict(arrowstyle="->", color=COL_ANNOT, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
    )


def load_required_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {name}: shape={df.shape}, columns={list(df.columns)}")
    return df


def load_optional_csv(path, name):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {name}: shape={df.shape}, columns={list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Optional file not found: {path}. Proceeding with None.")
        return None


def normalize_sweep(df):
    if df is None:
        return None
    out = df.copy()
    out["Benchmark_Canon"] = out["Benchmark"].map(canonical_benchmark)
    out["Size_MB"] = pd.to_numeric(out["Size_MB"], errors="coerce")
    out["Bandwidth_GBs"] = pd.to_numeric(out["Bandwidth_GBs"], errors="coerce")
    out["Stride_Bytes"] = pd.to_numeric(out["Stride_Bytes"], errors="coerce")
    return out


def normalize_power(df):
    if df is None:
        return None
    out = df.copy()
    out["Benchmark_Canon"] = out["Benchmark"].map(canonical_benchmark)
    num_cols = [
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
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalize_calib(df):
    if df is None:
        return None
    out = df.copy()
    out["Benchmark_Canon"] = out["Benchmark"].map(canonical_benchmark)
    out["Size_MB"] = pd.to_numeric(out["Size_MB"], errors="coerce")
    out["Iterations"] = pd.to_numeric(out["Iterations"], errors="coerce")
    out["Bandwidth_GBs"] = pd.to_numeric(out["Bandwidth_GBs"], errors="coerce")
    out["Stride_Bytes"] = pd.to_numeric(out["Stride_Bytes"], errors="coerce")
    return out


def save_figure(fig, out_dir, filename_base, saved_files):
    # Prior user preference: output PNG only.
    out_path = out_dir / f"{filename_base}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    saved_files.append(out_path.name)


def get_stride_color_map(strides):
    fixed = {
        8: "#1f77b4",
        16: "#ff7f0e",
        32: "#2ca02c",
        64: "#d62728",
        128: "#9467bd",
    }
    fallback = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    out = {}
    j = 0
    for s in sorted(strides):
        if s in fixed:
            out[s] = fixed[s]
        else:
            out[s] = fallback[j % len(fallback)]
            j += 1
    return out


def get_stride_style_map(strides):
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    out = {}
    for i, s in enumerate(sorted(strides)):
        out[s] = (styles[i % len(styles)], markers[i % len(markers)])
    return out


def split_sweep(df, bench):
    return df[df["Benchmark_Canon"] == bench].sort_values("Size_MB")


def split_power(df, bench):
    return df[df["Benchmark_Canon"] == bench].sort_values("Size_MB")


def merge_bw_pairs(df02, df03, bench):
    a = split_sweep(df02, bench)[["Size_MB", "Bandwidth_GBs"]].rename(columns={"Bandwidth_GBs": "BW_02"})
    b = split_sweep(df03, bench)[["Size_MB", "Bandwidth_GBs"]].rename(columns={"Bandwidth_GBs": "BW_03"})
    m = a.merge(b, on="Size_MB", how="inner").sort_values("Size_MB")
    return m


def mean_abs_pct_diff(df02, df03, bench):
    m = merge_bw_pairs(df02, df03, bench)
    m = m[m["BW_02"] > 0]
    if m.empty:
        return np.nan, np.nan
    pct = (m["BW_03"] - m["BW_02"]).abs() / m["BW_02"] * 100.0
    return float(pct.mean()), float(pct.max())


def level_mean_bw(sweep_df, bench, level):
    sub = split_sweep(sweep_df, bench).copy()
    sub["Level"] = sub["Size_MB"].apply(level_from_size)
    sel = sub[sub["Level"] == level]
    return float(sel["Bandwidth_GBs"].mean()) if not sel.empty else np.nan


def collapse_threshold_mb(store_df):
    dram = store_df[store_df["Size_MB"] > DR_BOUND_MB].sort_values("Size_MB")
    if dram.empty:
        return np.nan
    pre = dram[(dram["Size_MB"] >= 256) & (dram["Size_MB"] <= 4096)]["Bandwidth_GBs"]
    if pre.empty:
        pre = dram.iloc[: min(5, len(dram))]["Bandwidth_GBs"]
    pre_mean = float(pre.mean()) if not pre.empty else np.nan
    if np.isnan(pre_mean) or pre_mean <= 0:
        return np.nan
    cutoff = 0.7 * pre_mean
    drop = dram[dram["Bandwidth_GBs"] < cutoff]
    if drop.empty:
        return np.nan
    return float(drop.iloc[0]["Size_MB"])


def plot_fig01(sweep02_s32, sweep03_s32, out_dir, saved_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, bench, run03_col, title in [
        (axes[0], "Load", COL_LOAD, "(a) Load benchmark - run_02 vs run_03"),
        (axes[1], "Store", COL_STORE, "(b) Store benchmark - run_02 vs run_03"),
    ]:
        d02 = split_sweep(sweep02_s32, bench)
        d03 = split_sweep(sweep03_s32, bench)

        ax.plot(d02["Size_MB"], d02["Bandwidth_GBs"], color=COL_RUN02, linestyle="--",
                marker="o", alpha=0.7, label="run_02 (unlocked)")
        ax.plot(d03["Size_MB"], d03["Bandwidth_GBs"], color=run03_col, linestyle="-",
                marker="o", label="run_03 (clocks locked)")

        ax.set_xscale("log")
        all_x = pd.concat([d02["Size_MB"], d03["Size_MB"]]).dropna()
        if not all_x.empty:
            ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)
        apply_cache_shading(ax)

        mean_diff, max_diff = mean_abs_pct_diff(sweep02_s32, sweep03_s32, bench)
        print(f"Fig01 {bench}: mean abs pct diff={mean_diff:.2f}%, max diff={max_diff:.2f}%")
        ax.text(1.02, 0.90, f"Max difference: {max_diff:.1f}%", transform=ax.transAxes,
                ha="left", va="top", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

        ax.set_title(title)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, out_dir, "fig01_bw_run_comparison_stride32", saved_files)
    plt.close(fig)


def plot_fig02(sweep03_by_stride, out_dir, saved_files):
    available = sorted([s for s, df in sweep03_by_stride.items() if df is not None])
    if len(available) < 2:
        print("Skipping fig02: need at least two stride sweep files for run_03.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(available)
    colors_load = plt.cm.Blues(np.linspace(0.35, 0.9, n))
    colors_store = plt.cm.Reds(np.linspace(0.35, 0.9, n))
    style_map = get_stride_style_map(available)

    for idx, stride in enumerate(available):
        df = sweep03_by_stride[stride]
        dl = split_sweep(df, "Load")
        ds = split_sweep(df, "Store")
        ls, mk = style_map[stride]

        axes[0].plot(dl["Size_MB"], dl["Bandwidth_GBs"], color=colors_load[idx], linestyle=ls,
                     marker=mk, linewidth=1.6, label=f"stride={stride}B")
        axes[1].plot(ds["Size_MB"], ds["Bandwidth_GBs"], color=colors_store[idx], linestyle=ls,
                     marker=mk, linewidth=1.6, label=f"stride={stride}B")

    for ax in axes:
        ax.set_xscale("log")
        x_ref = pd.concat([split_sweep(sweep03_by_stride[available[0]], "Load")["Size_MB"],
                           split_sweep(sweep03_by_stride[available[0]], "Store")["Size_MB"]]).dropna()
        if not x_ref.empty:
            ax.set_xlim(x_ref.min() * 0.85, x_ref.max() * 1.15)
        apply_cache_shading(ax)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Bandwidth (GB/s)")

    axes[0].set_title("(a) Load - stride effect on bandwidth")
    axes[1].set_title("(b) Store - stride effect on bandwidth")

    ref_x = split_sweep(sweep03_by_stride[available[min(len(available)-1, 2)]], "Load")
    if not ref_x.empty:
        anchor = ref_x.iloc[min(len(ref_x)-1, len(ref_x)//2)]
        annotate_outside(
            axes[0],
            "Shorter stride = shorter chains = higher BW\n(pointer chain length = array_size / stride)",
            xy=(anchor["Size_MB"], anchor["Bandwidth_GBs"]),
            xytext_axes=(1.02, 0.72),
        )

    ref_s = split_sweep(sweep03_by_stride[available[min(len(available)-1, 2)]], "Store")
    if not ref_s.empty:
        anchor = ref_s.iloc[min(len(ref_s)-1, len(ref_s)//2)]
        annotate_outside(
            axes[1],
            "Smaller stride = more coalesced writes = higher BW\n(stride=8: 4 threads share one 32B sector)",
            xy=(anchor["Size_MB"], anchor["Bandwidth_GBs"]),
            xytext_axes=(1.02, 0.72),
        )

    axes[0].legend(title="Load stride", ncol=2, fontsize=9)
    axes[1].legend(title="Store stride", ncol=2, fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig02_bw_all_strides_run03", saved_files)
    plt.close(fig)


def plot_fig03(sweep03_by_stride, out_dir, saved_files):
    available = sorted([s for s, df in sweep03_by_stride.items() if df is not None])
    if len(available) < 2:
        print("Skipping fig03: need at least two stride sweep files for run_03.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    color_map = get_stride_color_map(available)
    style_map = get_stride_style_map(available)

    ratio_values = []
    for stride in available:
        df = sweep03_by_stride[stride]
        ld = split_sweep(df, "Load")[["Size_MB", "Bandwidth_GBs"]].rename(columns={"Bandwidth_GBs": "LD"})
        st = split_sweep(df, "Store")[["Size_MB", "Bandwidth_GBs"]].rename(columns={"Bandwidth_GBs": "ST"})
        m = st.merge(ld, on="Size_MB", how="inner")
        m = m[m["LD"] > 0].copy()
        if m.empty:
            continue
        m["Ratio"] = m["ST"] / m["LD"]
        ratio_values.append(m["Ratio"])
        ls, mk = style_map[stride]
        ax.plot(m["Size_MB"], m["Ratio"], color=color_map[stride], linestyle=ls,
                marker=mk, linewidth=1.6, label=f"stride={stride}B")

    if ratio_values:
        all_ratio = pd.concat(ratio_values)
        ymin = max(0.05, float(all_ratio.min()) * 0.8)
        ymax = float(all_ratio.max()) * 1.2
        ax.set_ylim(ymin, ymax)
        ax.axhspan(1.0, ymax, color="#EAF8EA", alpha=0.35, zorder=0)
        ax.axhspan(ymin, 1.0, color="#FDECEC", alpha=0.35, zorder=0)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.text(0.98, 1.0, "Store = Load", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9)

    ax.set_xscale("log")
    x_ref = split_sweep(sweep03_by_stride[available[0]], "Load")["Size_MB"].dropna()
    if not x_ref.empty:
        ax.set_xlim(x_ref.min() * 0.85, x_ref.max() * 1.15)

    ax.axvline(L1_BOUND_MB, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(DR_BOUND_MB, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title("Store/Load Bandwidth Ratio vs Array Size - all strides, run_03")
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Store BW / Load BW")

    if ratio_values:
        anchor_df = split_sweep(sweep03_by_stride[32], "Store") if 32 in sweep03_by_stride else split_sweep(sweep03_by_stride[available[0]], "Store")
        if not anchor_df.empty:
            anchor = anchor_df.iloc[min(len(anchor_df) - 1, len(anchor_df) // 2)]
            annotate_outside(
                ax,
                "Load faster in L2 at stride=32 due to pointer chain\n"
                "length advantage. Store faster everywhere at stride=8\n"
                "due to coalescence.",
                xy=(anchor["Size_MB"], 1.0),
                xytext_axes=(1.02, 0.58),
            )

    ax.legend(title="Stride", ncol=2, fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig03_store_load_ratio", saved_files)
    plt.close(fig)


def plot_fig04(sweep02_s32, sweep03_s32, sweep02_s8, sweep03_s8, out_dir, saved_files):
    fig, ax = plt.subplots(figsize=(10, 5))

    st02_32 = split_sweep(sweep02_s32, "Store")
    st03_32 = split_sweep(sweep03_s32, "Store")
    st02_32 = st02_32[st02_32["Size_MB"] > DR_BOUND_MB].sort_values("Size_MB")
    st03_32 = st03_32[st03_32["Size_MB"] > DR_BOUND_MB].sort_values("Size_MB")

    ax.plot(st02_32["Size_MB"], st02_32["Bandwidth_GBs"], color=COL_RUN02,
            linestyle="--", marker="s", linewidth=1.7, label="run_02 stride=32")
    ax.plot(st03_32["Size_MB"], st03_32["Bandwidth_GBs"], color=COL_STORE,
            linestyle="-", marker="s", linewidth=1.9, label="run_03 stride=32")

    if sweep02_s8 is not None:
        st02_8 = split_sweep(sweep02_s8, "Store")
        st02_8 = st02_8[st02_8["Size_MB"] > DR_BOUND_MB].sort_values("Size_MB")
        ax.plot(st02_8["Size_MB"], st02_8["Bandwidth_GBs"], color=COL_RUN02,
                linestyle=":", marker="o", linewidth=1.5, label="run_02 stride=8")
    if sweep03_s8 is not None:
        st03_8 = split_sweep(sweep03_s8, "Store")
        st03_8 = st03_8[st03_8["Size_MB"] > DR_BOUND_MB].sort_values("Size_MB")
        ax.plot(st03_8["Size_MB"], st03_8["Bandwidth_GBs"], color=COL_STORE,
                linestyle="--", marker="o", linewidth=1.5, label="run_03 stride=8")

    all_x = [st02_32["Size_MB"], st03_32["Size_MB"]]
    if sweep02_s8 is not None:
        all_x.append(st02_8["Size_MB"])
    if sweep03_s8 is not None:
        all_x.append(st03_8["Size_MB"])
    all_x = pd.concat(all_x).dropna()
    if not all_x.empty:
        ax.set_xlim(all_x.min() * 0.95, all_x.max() * 1.03)

    ax.axvline(8192, color="gray", linestyle="--", linewidth=1.0)
    ax.text(8192 * 1.002, ax.get_ylim()[1] * 0.95, "~8GB: collapse threshold",
            color="gray", ha="left", va="top", fontsize=9)

    pre = st03_32[(st03_32["Size_MB"] >= 256) & (st03_32["Size_MB"] <= 4096)]["Bandwidth_GBs"]
    post = st03_32[st03_32["Size_MB"] >= 12288]["Bandwidth_GBs"]
    if not pre.empty and not post.empty:
        pre_mean = float(pre.mean())
        post_mean = float(post.mean())
        pct = 100.0 * (post_mean - pre_mean) / pre_mean if pre_mean > 0 else np.nan
        print(f"Fig04 run_03 stride32 collapse: {pre_mean:.1f} -> {post_mean:.1f} GB/s ({pct:.1f}%)")
        target_x = float(st03_32[st03_32["Size_MB"] >= 12288].iloc[0]["Size_MB"])
        target_y = float(st03_32[st03_32["Size_MB"] >= 12288].iloc[0]["Bandwidth_GBs"])
        annotate_outside(
            ax,
            f"{pre_mean:.0f} -> {post_mean:.0f} GB/s\n({pct:.0f}%)",
            xy=(target_x, target_y),
            xytext_axes=(1.02, 0.50),
        )

    ax.set_title("Store Bandwidth Collapse at Large DRAM Sizes - RTX 5000 Ada")
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Store Bandwidth (GB/s)")
    ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, out_dir, "fig04_store_dram_collapse", saved_files)
    plt.close(fig)


def plot_fig05(power02_s32, power03_s32, out_dir, saved_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, bench, run03_col, title in [
        (axes[0], "Load", COL_LOAD, "(a) Load - kernel power, run_02 vs run_03"),
        (axes[1], "Store", COL_STORE, "(b) Store - kernel power, run_02 vs run_03"),
    ]:
        d02 = split_power(power02_s32, bench)
        d03 = split_power(power03_s32, bench)

        ax.plot(d02["Size_MB"], d02["Avg_Kernel_Power_W"], color=COL_RUN02, linestyle="--",
                marker="o", alpha=0.8, label="run_02 kernel")
        ax.plot(d03["Size_MB"], d03["Avg_Kernel_Power_W"], color=run03_col, linestyle="-",
                marker="o", label="run_03 kernel")

        ax.scatter(d02["Size_MB"], d02["Static_Power_W"], color=COL_RUN02, marker="x", s=34,
                   label="run_02 static")
        ax.scatter(d03["Size_MB"], d03["Static_Power_W"], color=run03_col, marker="x", s=34,
                   label="run_03 static")

        run03_static_mean = float(d03["Static_Power_W"].mean()) if not d03.empty else np.nan
        ax.axhline(run03_static_mean, color=run03_col, linestyle="-.", linewidth=1.0,
                   alpha=0.8, label="run_03 static mean")

        ax.set_xscale("log")
        all_x = pd.concat([d02["Size_MB"], d03["Size_MB"]]).dropna()
        if not all_x.empty:
            ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)
        apply_cache_shading(ax)

        ax.set_title(title)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Power (W)")

    st03 = split_power(power03_s32, "Store")
    st03_dram = st03[st03["Size_MB"] > DR_BOUND_MB]
    ld03_dram = split_power(power03_s32, "Load")
    ld03_dram = ld03_dram[ld03_dram["Size_MB"] > DR_BOUND_MB]
    if not st03_dram.empty and not ld03_dram.empty:
        s_min, s_max = float(st03_dram["Avg_Kernel_Power_W"].min()), float(st03_dram["Avg_Kernel_Power_W"].max())
        l_mean = float(ld03_dram["Avg_Kernel_Power_W"].mean())
        print(f"Fig05 DRAM power run_03: store={s_min:.1f}-{s_max:.1f}W, load mean={l_mean:.1f}W")
        anchor = st03_dram.loc[st03_dram["Avg_Kernel_Power_W"].idxmax()]
        annotate_outside(
            axes[1],
            f"Store power: {s_min:.0f}-{s_max:.0f}W\n(vs ~{l_mean:.0f}W for load)",
            xy=(anchor["Size_MB"], anchor["Avg_Kernel_Power_W"]),
            xytext_axes=(1.02, 0.75),
        )

    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig05_power_comparison", saved_files)
    plt.close(fig)


def plot_fig06(power02_s32, power03_s32, out_dir, saved_files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    p02 = filter_nonzero_energy(power02_s32)
    p03 = filter_nonzero_energy(power03_s32)

    for ax, bench, run03_col, title in [
        (axes[0], "Load", COL_LOAD, "(a) Load - dynamic energy (valid points only)"),
        (axes[1], "Store", COL_STORE, "(b) Store - dynamic energy (valid points only)"),
    ]:
        d02 = split_power(p02, bench)
        d03 = split_power(p03, bench)

        y02 = d02["Mean_Dynamic_Energy_J"] * 1000.0
        y03 = d03["Mean_Dynamic_Energy_J"] * 1000.0
        e02 = d02["Std_Dynamic_Energy_J"] * 1000.0
        e03 = d03["Std_Dynamic_Energy_J"] * 1000.0

        ax.plot(d02["Size_MB"], y02, color=COL_RUN02, linestyle="--", marker="o", label="run_02")
        ax.plot(d03["Size_MB"], y03, color=run03_col, linestyle="-", marker="o", label="run_03")

        mask02 = e02 > 0
        mask03 = e03 > 0
        if mask02.any():
            ax.errorbar(d02.loc[mask02, "Size_MB"], y02.loc[mask02], yerr=e02.loc[mask02],
                        fmt="none", ecolor=COL_RUN02, capsize=3, linewidth=1.0)
        if mask03.any():
            ax.errorbar(d03.loc[mask03, "Size_MB"], y03.loc[mask03], yerr=e03.loc[mask03],
                        fmt="none", ecolor=run03_col, capsize=3, linewidth=1.0)

        if not d02.empty or not d03.empty:
            x_all = pd.concat([d02["Size_MB"], d03["Size_MB"]]).dropna()
            y_all = pd.concat([y02, y03]).replace([np.inf, -np.inf], np.nan).dropna()
            if not x_all.empty:
                ax.set_xlim(x_all.min() * 0.85, x_all.max() * 1.15)
            if not y_all.empty:
                ax.set_ylim(y_all.min() * 0.8, y_all.max() * 1.25)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Dynamic Energy (mJ)")

        n02 = len(d02)
        n03 = len(d03)
        print(f"Fig06 {bench}: run_02 valid points={n02}, run_03 valid points={n03}")
        ax.text(1.02, 0.16, f"run_02: {n02} valid points\nrun_03: {n03} valid points",
                transform=ax.transAxes, ha="left", va="bottom", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, out_dir, "fig06_dynamic_energy_comparison", saved_files)
    plt.close(fig)


def plot_fig07(power02_s32, power03_s32, out_dir, saved_files):
    fig, ax = plt.subplots(figsize=(10, 5))

    l02 = split_power(power02_s32, "Load")
    l03 = split_power(power03_s32, "Load")
    s02 = split_power(power02_s32, "Store")
    s03 = split_power(power03_s32, "Store")

    ax.plot(l02["Size_MB"], l02["Static_Power_W"], color=COL_RUN02, linestyle="--", marker="o", label="run_02 load static")
    ax.plot(l03["Size_MB"], l03["Static_Power_W"], color=COL_LOAD, linestyle="-", marker="o", label="run_03 load static")
    ax.plot(s02["Size_MB"], s02["Static_Power_W"], color=COL_RUN02, linestyle="--", marker="s", label="run_02 store static")
    ax.plot(s03["Size_MB"], s03["Static_Power_W"], color=COL_STORE, linestyle="-", marker="s", label="run_03 store static")

    mean02 = float(power02_s32["Static_Power_W"].mean())
    mean03 = float(power03_s32["Static_Power_W"].mean())
    std02 = float(power02_s32["Static_Power_W"].std(ddof=1))
    std03 = float(power03_s32["Static_Power_W"].std(ddof=1))
    print(f"Fig07 static baseline: run_02 mean={mean02:.2f}W sigma={std02:.2f}W | run_03 mean={mean03:.2f}W sigma={std03:.2f}W")

    ax.axhline(mean02, color=COL_RUN02, linestyle="--", linewidth=1.0, label="run_02 mean")
    ax.axhline(mean03, color=COL_RUN03, linestyle="-", linewidth=1.2, label="run_03 mean")

    ax.set_xscale("log")
    all_x = pd.concat([l02["Size_MB"], l03["Size_MB"], s02["Size_MB"], s03["Size_MB"]]).dropna()
    if not all_x.empty:
        ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.15)

    ax.set_title("Static (Idle) Power Baseline - run_02 vs run_03, stride=32")
    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Static Power (W)")

    ax.text(1.02, 0.78, f"run_02 sigma = {std02:.2f} W\nrun_03 sigma = {std03:.2f} W",
            transform=ax.transAxes, ha="left", va="top", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig07_static_power_baseline", saved_files)
    plt.close(fig)


def plot_fig08(calib02, calib03, out_dir, saved_files):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)

    runs = [("run_02", calib02), ("run_03", calib03)]
    levels = ["L1 Cache", "L2 Cache", "DRAM"]

    for r_idx, (run_name, cdf) in enumerate(runs):
        for l_idx, level in enumerate(levels):
            ax = axes[r_idx, l_idx]
            sub = cdf[cdf["Level"] == level].copy()
            ld = sub[sub["Benchmark_Canon"] == "Load"].sort_values("Iterations")
            st = sub[sub["Benchmark_Canon"] == "Store"].sort_values("Iterations")

            ax.plot(ld["Iterations"], ld["Bandwidth_GBs"], color=COL_LOAD, marker="o", linestyle="-", label="Load")
            ax.plot(st["Iterations"], st["Bandwidth_GBs"], color=COL_STORE, marker="s", linestyle="-", label="Store")

            ld_mean = float(ld["Bandwidth_GBs"].mean()) if not ld.empty else np.nan
            ld_std = float(ld["Bandwidth_GBs"].std(ddof=1)) if len(ld) > 1 else 0.0

            st_stats = st
            if level == "DRAM":
                out = st[st["Iterations"] == 1]
                if not out.empty:
                    ox = float(out.iloc[0]["Iterations"])
                    oy = float(out.iloc[0]["Bandwidth_GBs"])
                    ax.scatter([ox], [oy], color=COL_STORE, marker="X", s=90, zorder=5)
                    annotate_outside(ax, "warmup outlier", xy=(ox, oy), xytext_axes=(1.02, 0.30))
                st_stats = st[st["Iterations"] != 1]

            st_mean = float(st_stats["Bandwidth_GBs"].mean()) if not st_stats.empty else np.nan
            st_std = float(st_stats["Bandwidth_GBs"].std(ddof=1)) if len(st_stats) > 1 else 0.0

            if not np.isnan(ld_mean):
                ax.axhline(ld_mean, color=COL_LOAD, linestyle="--", linewidth=0.9, alpha=0.8)
            if not np.isnan(st_mean):
                ax.axhline(st_mean, color=COL_STORE, linestyle="--", linewidth=0.9, alpha=0.8)

            print(f"Fig08 {run_name} {level}: Load={ld_mean:.2f}+/-{ld_std:.2f}, Store={st_mean:.2f}+/-{st_std:.2f}")
            ax.text(0.02, -0.34,
                    f"Load: {ld_mean:.1f}+/-{ld_std:.1f}\nStore: {st_mean:.1f}+/-{st_std:.1f}",
                    transform=ax.transAxes, ha="left", va="top", clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9), fontsize=8)

            ax.set_xscale("log")
            ax.set_title(f"{run_name} - {level.replace(' Cache', '')}")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Bandwidth (GB/s)")
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Calibration - bandwidth stability, run_02 vs run_03")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    save_figure(fig, out_dir, "fig08_calibration_comparison", saved_files)
    plt.close(fig)


def plot_fig09(power03_by_stride, out_dir, saved_files):
    available = sorted([s for s, df in power03_by_stride.items() if df is not None])
    if len(available) < 2:
        print("Skipping fig09: need at least two power stride files for run_03.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    levels = ["L1", "L2", "DRAM"]
    titles = ["(a) L1", "(b) L2", "(c) DRAM"]

    for i, level in enumerate(levels):
        ax = axes[i]
        x_load, y_load, x_store, y_store = [], [], [], []

        for stride in available:
            df = filter_nonzero_energy(power03_by_stride[stride])
            df = df[df["Dynamic_Energy_pJ_bit"] > 0].copy()
            if df.empty:
                continue
            df["Level"] = df["Size_MB"].apply(level_from_size)
            lv = df[df["Level"] == level]
            ld = lv[lv["Benchmark_Canon"] == "Load"]
            st = lv[lv["Benchmark_Canon"] == "Store"]

            if not ld.empty:
                x_load.append(stride)
                y_load.append(float(ld["Dynamic_Energy_pJ_bit"].mean()))
            if not st.empty:
                x_store.append(stride)
                y_store.append(float(st["Dynamic_Energy_pJ_bit"].mean()))

        if len(set(x_load + x_store)) < 2:
            ax.text(0.5, 0.5, "Insufficient valid measurements",
                    transform=ax.transAxes, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))
        else:
            if len(x_load) >= 2:
                order = np.argsort(x_load)
                xl = np.array(x_load)[order]
                yl = np.array(y_load)[order]
                ax.plot(xl, yl, color=COL_LOAD, marker="o", linewidth=1.8, label="Load")
            if len(x_store) >= 2:
                order = np.argsort(x_store)
                xs = np.array(x_store)[order]
                ys = np.array(y_store)[order]
                ax.plot(xs, ys, color=COL_STORE, marker="s", linewidth=1.8, label="Store")

            ax.axvline(32, color="gray", linestyle="--", linewidth=1.0)
            ax.text(32 + 1.2, ax.get_ylim()[1] * 0.95, "1 sector", color="gray", fontsize=8,
                    ha="left", va="top")

        ax.set_title(titles[i])
        ax.set_xlabel("Stride_Bytes")
        ax.set_ylabel("Dynamic Energy (pJ/bit)")
        if len(set(x_load + x_store)) >= 2:
            ax.set_yscale("log")
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("Stride saturation curve - energy per bit vs stride, run_03")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, out_dir, "fig09_stride_saturation", saved_files)
    plt.close(fig)


def fmt_num(x):
    if np.isnan(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def plot_fig10_summary_table(sweep02_s32, sweep03_s32, power02_s32, power03_s32, out_dir, saved_files):
    metrics = []

    for level in ["L1", "L2", "DRAM"]:
        metrics.append((f"Load BW - {level} (GB/s)",
                        level_mean_bw(sweep02_s32, "Load", level),
                        level_mean_bw(sweep03_s32, "Load", level)))
        metrics.append((f"Store BW - {level} (GB/s)",
                        level_mean_bw(sweep02_s32, "Store", level),
                        level_mean_bw(sweep03_s32, "Store", level)))

    for level in ["L1", "L2", "DRAM"]:
        ld02 = level_mean_bw(sweep02_s32, "Load", level)
        st02 = level_mean_bw(sweep02_s32, "Store", level)
        ld03 = level_mean_bw(sweep03_s32, "Load", level)
        st03 = level_mean_bw(sweep03_s32, "Store", level)
        r02 = st02 / ld02 if ld02 and not np.isnan(ld02) else np.nan
        r03 = st03 / ld03 if ld03 and not np.isnan(ld03) else np.nan
        metrics.append((f"Store/Load BW ratio - {level}", r02, r03))

    st02 = split_sweep(sweep02_s32, "Store")
    st03 = split_sweep(sweep03_s32, "Store")
    thr02 = collapse_threshold_mb(st02)
    thr03 = collapse_threshold_mb(st03)
    metrics.append(("Store collapse threshold (MB)", thr02, thr03))

    p02_store_d = split_power(power02_s32, "Store")
    p03_store_d = split_power(power03_s32, "Store")
    p02_store_d = p02_store_d[p02_store_d["Size_MB"] > DR_BOUND_MB]
    p03_store_d = p03_store_d[p03_store_d["Size_MB"] > DR_BOUND_MB]
    peak_store02 = float(p02_store_d["Avg_Kernel_Power_W"].max()) if not p02_store_d.empty else np.nan
    peak_store03 = float(p03_store_d["Avg_Kernel_Power_W"].max()) if not p03_store_d.empty else np.nan
    metrics.append(("Peak store power - DRAM (W)", peak_store02, peak_store03))

    p02_load_d = split_power(power02_s32, "Load")
    p03_load_d = split_power(power03_s32, "Load")
    p02_load_d = p02_load_d[p02_load_d["Size_MB"] > DR_BOUND_MB]
    p03_load_d = p03_load_d[p03_load_d["Size_MB"] > DR_BOUND_MB]
    load_pow02 = float(p02_load_d["Avg_Kernel_Power_W"].mean()) if not p02_load_d.empty else np.nan
    load_pow03 = float(p03_load_d["Avg_Kernel_Power_W"].mean()) if not p03_load_d.empty else np.nan
    metrics.append(("Load power - DRAM (W)", load_pow02, load_pow03))

    rows = []
    diffs = []
    for name, v02, v03 in metrics:
        if np.isnan(v02) or v02 == 0 or np.isnan(v03):
            d = np.nan
        else:
            d = 100.0 * (v03 - v02) / v02
        rows.append([fmt_num(v02), fmt_num(v03), "NA" if np.isnan(d) else f"{d:+.1f}%"])
        diffs.append(d)
        print(f"Fig10 metric | {name}: run_02={v02}, run_03={v03}, diff={d}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        rowLabels=[m[0] for m in metrics],
        colLabels=["run_02 (unlocked clocks)", "run_03 (locked clocks)", "Difference (%)"],
        loc="center",
        cellLoc="center",
        rowLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    # Color-code Difference column
    for i, d in enumerate(diffs, start=1):
        cell = table[(i, 2)]
        if not np.isnan(d) and abs(d) > 5.0:
            cell.set_facecolor("#DFF4DF")
        else:
            cell.set_facecolor("#EEEEEE")

    ax.set_title("Key Metrics Summary - run_02 vs run_03, stride=32")

    save_figure(fig, out_dir, "fig10_summary_table", saved_files)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate run_02 vs run_03 analysis figures.")
    parser.add_argument("--run02", type=str, required=True)
    parser.add_argument("--run03", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    run02_dir = Path(args.run02)
    run03_dir = Path(args.run03)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Required files
    sweep02_s32 = normalize_sweep(load_required_csv(run02_dir / "sweep_stride32.csv", "run02 sweep_stride32.csv"))
    sweep02_s8 = normalize_sweep(load_required_csv(run02_dir / "sweep_stride8.csv", "run02 sweep_stride8.csv"))
    power02_s32 = normalize_power(load_required_csv(run02_dir / "power_stride32.csv", "run02 power_stride32.csv"))
    calib02 = normalize_calib(load_required_csv(run02_dir / "calibrate_stride32.csv", "run02 calibrate_stride32.csv"))

    sweep03_s32 = normalize_sweep(load_required_csv(run03_dir / "sweep_stride32.csv", "run03 sweep_stride32.csv"))
    sweep03_s8 = normalize_sweep(load_required_csv(run03_dir / "sweep_stride8.csv", "run03 sweep_stride8.csv"))
    power03_s32 = normalize_power(load_required_csv(run03_dir / "power_stride32.csv", "run03 power_stride32.csv"))
    calib03 = normalize_calib(load_required_csv(run03_dir / "calibrate_stride32.csv", "run03 calibrate_stride32.csv"))

    # Optional files
    sweep02_s16 = normalize_sweep(load_optional_csv(run02_dir / "sweep_stride16.csv", "run02 sweep_stride16.csv"))
    sweep02_s64 = normalize_sweep(load_optional_csv(run02_dir / "sweep_stride64.csv", "run02 sweep_stride64.csv"))
    sweep02_s128 = normalize_sweep(load_optional_csv(run02_dir / "sweep_stride128.csv", "run02 sweep_stride128.csv"))
    power02_s8 = normalize_power(load_optional_csv(run02_dir / "power_stride8.csv", "run02 power_stride8.csv"))

    sweep03_s16 = normalize_sweep(load_optional_csv(run03_dir / "sweep_stride16.csv", "run03 sweep_stride16.csv"))
    sweep03_s64 = normalize_sweep(load_optional_csv(run03_dir / "sweep_stride64.csv", "run03 sweep_stride64.csv"))
    sweep03_s128 = normalize_sweep(load_optional_csv(run03_dir / "sweep_stride128.csv", "run03 sweep_stride128.csv"))
    power03_s8 = normalize_power(load_optional_csv(run03_dir / "power_stride8.csv", "run03 power_stride8.csv"))

    power02_s16 = normalize_power(load_optional_csv(run02_dir / "power_stride16.csv", "run02 power_stride16.csv"))
    power02_s64 = normalize_power(load_optional_csv(run02_dir / "power_stride64.csv", "run02 power_stride64.csv"))
    power02_s128 = normalize_power(load_optional_csv(run02_dir / "power_stride128.csv", "run02 power_stride128.csv"))

    power03_s16 = normalize_power(load_optional_csv(run03_dir / "power_stride16.csv", "run03 power_stride16.csv"))
    power03_s64 = normalize_power(load_optional_csv(run03_dir / "power_stride64.csv", "run03 power_stride64.csv"))
    power03_s128 = normalize_power(load_optional_csv(run03_dir / "power_stride128.csv", "run03 power_stride128.csv"))

    sweep03_by_stride = {
        8: sweep03_s8,
        16: sweep03_s16,
        32: sweep03_s32,
        64: sweep03_s64,
        128: sweep03_s128,
    }

    power03_by_stride = {
        8: power03_s8,
        16: power03_s16,
        32: power03_s32,
        64: power03_s64,
        128: power03_s128,
    }

    saved_files = []

    fig_jobs = [
        ("fig01", lambda: plot_fig01(sweep02_s32, sweep03_s32, out_dir, saved_files)),
        ("fig02", lambda: plot_fig02(sweep03_by_stride, out_dir, saved_files)),
        ("fig03", lambda: plot_fig03(sweep03_by_stride, out_dir, saved_files)),
        ("fig04", lambda: plot_fig04(sweep02_s32, sweep03_s32, sweep02_s8, sweep03_s8, out_dir, saved_files)),
        ("fig05", lambda: plot_fig05(power02_s32, power03_s32, out_dir, saved_files)),
        ("fig06", lambda: plot_fig06(power02_s32, power03_s32, out_dir, saved_files)),
        ("fig07", lambda: plot_fig07(power02_s32, power03_s32, out_dir, saved_files)),
        ("fig08", lambda: plot_fig08(calib02, calib03, out_dir, saved_files)),
        ("fig09", lambda: plot_fig09(power03_by_stride, out_dir, saved_files)),
        ("fig10", lambda: plot_fig10_summary_table(sweep02_s32, sweep03_s32, power02_s32, power03_s32, out_dir, saved_files)),
    ]

    for name, fn in fig_jobs:
        try:
            print(f"\n--- Generating {name} ---")
            fn()
        except Exception as exc:
            print(f"Error while generating {name}: {exc}")
            traceback.print_exc()

    print("\nFinished. Successfully saved figures:")
    for f in saved_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
