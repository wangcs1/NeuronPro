# analysis/psth_fano.py
"""
psth_fano.py
------------
从方向编码数据集中分析：
- tuning curve（平均 firing rate vs 方向）
- PSTH（peristimulus time histogram）
- Fano factor（trial-to-trial variability）

要求数据格式：
npz 文件中至少包含：
- spikes: (n_examples, T_steps, n_neurons) 0/1
- labels: (n_examples,)
- directions_deg: (n_dirs,)
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(dataset_path: str):
    data = np.load(dataset_path, allow_pickle=True)
    spikes = data["spikes"]            # (n_examples, T_steps, n_neurons)
    labels = data["labels"]            # (n_examples,)
    directions_deg = data["directions_deg"]  # (n_dirs,)
    T_steps = spikes.shape[1]
    return spikes, labels, directions_deg, T_steps


def compute_tuning(
    spikes: np.ndarray,
    labels: np.ndarray,
    directions_deg: np.ndarray,
    neuron_idx: int,
    T_steps: int,
    dt: float = 1.0,
):
    """
    计算某个 neuron 的 tuning curve（平均 firing rate vs 方向）。

    返回：
    - dir_list: 刺激方向（deg）
    - mean_rates: 每个方向上的平均 firing rate（Hz）
    """
    dir_list = directions_deg
    mean_rates = []

    T_ms = T_steps * dt
    T_sec = T_ms / 1000.0

    for d_idx, direction in enumerate(dir_list):
        sel = labels == d_idx              # 选择这个方向的 trial
        sp_dir = spikes[sel, :, neuron_idx]  # (n_trials, T_steps)
        counts = sp_dir.sum(axis=1)        # 每个 trial 的总 spikes
        mean_rate = counts.mean() / T_sec  # Hz
        mean_rates.append(mean_rate)

    mean_rates = np.asarray(mean_rates)
    return dir_list, mean_rates


def compute_psth(
    spikes: np.ndarray,
    labels: np.ndarray,
    directions_deg: np.ndarray,
    neuron_idx: int,
    dir_index: int,
    dt: float = 1.0,
):
    """
    计算某个 neuron 在指定方向下的 PSTH（时间上平均 firing rate）。

    返回：
    - time_bins_ms: 每个时间 bin 的中心（ms）
    - psth_rate: 每个时间 bin 的平均 firing rate（Hz）
    """
    sel = labels == dir_index
    sp_sel = spikes[sel, :, neuron_idx]  # (n_trials, T_steps)
    # 每个时间点上，对 trial 求平均
    mean_spikes_per_bin = sp_sel.mean(axis=0)   # (T_steps,)
    # 0/1 代表每 dt ms 的 spike 概率 → Hz：p / (dt/1000)
    psth_rate = mean_spikes_per_bin / (dt / 1000.0)  # (T_steps,)
    T_steps = sp_sel.shape[1]
    time_bins_ms = np.arange(T_steps) * dt
    return time_bins_ms, psth_rate


def compute_fano(
    spikes: np.ndarray,
    labels: np.ndarray,
    directions_deg: np.ndarray,
    dt: float = 1.0,
):
    """
    计算所有 neuron 在所有方向上的 Fano factor。

    做法：
    - 对每个 neuron、每个方向：
      * 统计每个 trial 的 spike count
      * 计算 mean, var
      * Fano = var / mean（忽略 mean 为 0 的情况）
    - 汇总成一个 list，返回所有有效 Fano 值
    """
    n_examples, T_steps, n_neurons = spikes.shape
    T_sec = (T_steps * dt) / 1000.0

    fano_values = []

    for neuron_idx in range(n_neurons):
        for d_idx, direction in enumerate(directions_deg):
            sel = labels == d_idx
            sp_sel = spikes[sel, :, neuron_idx]  # (n_trials, T_steps)
            counts = sp_sel.sum(axis=1)
            mean_c = counts.mean()
            var_c = counts.var(ddof=1)
            if mean_c > 1e-6:  # 避免除以 0
                fano = var_c / mean_c
                fano_values.append(fano)

    fano_values = np.asarray(fano_values)
    return fano_values


def analyze_and_plot(
    dataset_path: str,
    neuron_idx: int = 0,
    dir_index_for_psth: int | None = None,
    dt: float = 1.0,
):
    spikes, labels, directions_deg, T_steps = load_dataset(dataset_path)
    n_examples, _, n_neurons = spikes.shape

    if neuron_idx < 0 or neuron_idx >= n_neurons:
        raise ValueError(f"neuron_idx {neuron_idx} out of range [0, {n_neurons-1}]")

    if dir_index_for_psth is None:
        dir_index_for_psth = 0
    if dir_index_for_psth < 0 or dir_index_for_psth >= len(directions_deg):
        raise ValueError(
            f"dir_index_for_psth {dir_index_for_psth} out of range [0, {len(directions_deg)-1}]"
        )

    print(f"Dataset: {dataset_path}")
    print(f"spikes shape: {spikes.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"directions_deg: {directions_deg}")
    print(f"Analyzing neuron {neuron_idx}, PSTH direction index {dir_index_for_psth} "
          f"({directions_deg[dir_index_for_psth]} deg)")

    # ---- 1) tuning curve ----
    dir_list, mean_rates = compute_tuning(
        spikes, labels, directions_deg, neuron_idx=neuron_idx,
        T_steps=T_steps, dt=dt,
    )

    plt.figure(figsize=(5, 4))
    plt.plot(dir_list, mean_rates, marker="o")
    plt.xlabel("Direction (deg)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title(f"Neuron {neuron_idx} tuning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- 2) PSTH ----
    time_bins_ms, psth_rate = compute_psth(
        spikes, labels, directions_deg,
        neuron_idx=neuron_idx, dir_index=dir_index_for_psth, dt=dt,
    )

    plt.figure(figsize=(5, 4))
    plt.plot(time_bins_ms, psth_rate)
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(
        f"Neuron {neuron_idx} PSTH | direction={directions_deg[dir_index_for_psth]} deg"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- 3) Fano factor 分布 ----
    fano_values = compute_fano(spikes, labels, directions_deg, dt=dt)
    print(f"Fano values: mean={fano_values.mean():.3f}, "
          f"std={fano_values.std():.3f}, "
          f"min={fano_values.min():.3f}, max={fano_values.max():.3f}")

    plt.figure(figsize=(5, 4))
    plt.hist(fano_values, bins=20, alpha=0.8, edgecolor="black")
    plt.xlabel("Fano factor")
    plt.ylabel("Count")
    plt.title("Distribution of Fano factors (all neurons, all directions)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="direction_encoding_static8.npz",
        help="Path to dataset npz file",
    )
    parser.add_argument(
        "--neuron",
        type=int,
        default=0,
        help="Neuron index for tuning/PSTH plot",
    )
    parser.add_argument(
        "--dir_index",
        type=int,
        default=0,
        help="Direction index for PSTH (0..n_dirs-1)",
    )
    args = parser.parse_args()

    analyze_and_plot(
        dataset_path=args.data,
        neuron_idx=args.neuron,
        dir_index_for_psth=args.dir_index,
        dt=1.0,
    )


if __name__ == "__main__":
    main()
