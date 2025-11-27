# encoding/dataset_builder.py
"""
dataset_builder.py
------------------
高层接口：基于 tuning curve + Poisson spike train 生成
视觉方向编码数据集 (spikes, labels)。

这个模块不涉及具体的解码器（MLP/SNN），只负责「刺激 → spike」。
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from encoding.tuning import (
    generate_preferred_directions,
    sample_r_max,
    direction_tuning_gaussian,
)
from encoding.poisson_spike import poisson_population_spikes



def generate_direction_encoding_dataset(
    directions_deg: np.ndarray,
    n_neurons: int,
    trials_per_dir: int = 100,
    T: float = 400.0,
    dt: float = 1.0,
    r_baseline: float = 5.0,
    r_max_mean: float = 20.0,
    r_max_std: float = 5.0,
    tuning_sigma_deg: float = 40.0,
    jitter_pref_deg: float = 5.0,
    # 噪声参数
    gain_sigma: float = 0.0,
    shared_std: float = 0.0,
    indep_std: float = 0.0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    生成一个“方向编码”数据集：
        给定一组方向 θ_k，若干神经元（有 preferred direction、tuning 曲线），
        使用 Poisson spike 生成 spike train。

    返回的字典可以直接保存为 npz 或喂给解码器训练。

    参数
    ----
    directions_deg : array-like, shape (n_dirs,)
        刺激方向（度），例如 np.arange(0, 360, 45)
    n_neurons : int
        神经元数量
    trials_per_dir : int
        每个方向重复的 trial 数
    T : float
        每个 trial 的时长（ms）
    dt : float
        时间步长（ms）
    r_baseline : float
        baseline firing rate（Hz）
    r_max_mean, r_max_std : float
        调制幅度 r_max 的高斯采样参数（Hz）
    tuning_sigma_deg : float
        tuning 曲线宽度（度）
    jitter_pref_deg : float
        preferred direction 的高斯抖动（度）
    gain_sigma : float
        trial-by-trial multiplicative gain 的 log-normal σ。
        =0 表示不加 gain noise。
    shared_std : float
        每个 trial 一次的 shared additive noise 标准差（Hz）。
        =0 表示不加。
    indep_std : float
        每个 trial、每个 neuron 的 independent additive noise 标准差（Hz）。
        =0 表示不加。
    seed : int | None
        随机种子

    返回
    ----
    out : dict
        包含以下字段：
        - spikes: ndarray, (n_examples, T_steps, n_neurons), 0/1
        - labels: ndarray, (n_examples,) 方向类别索引 0..n_dirs-1
        - directions_deg: ndarray, (n_dirs,)
        - theta_prefs: ndarray, (n_neurons,) preferred directions
        - r_max: ndarray, (n_neurons,)
        - meta: dict, 记录上面这些参数，方便保存/分析
    """
    rng = np.random.default_rng(seed)
    directions_deg = np.asarray(directions_deg)
    n_dirs = directions_deg.shape[0]
    T_steps = int(T / dt)

    # 1) 构建 population：preferred directions + r_max
    theta_prefs = generate_preferred_directions(
        n_neurons=n_neurons,
        jitter_deg=jitter_pref_deg,
        seed=rng.integers(1_000_000_000),
    )
    r_max = sample_r_max(
        n_neurons=n_neurons,
        r_max_mean=r_max_mean,
        r_max_std=r_max_std,
        min_rate=1.0,
        max_rate=None,
        seed=rng.integers(1_000_000_000),
    )

    # 2) 计算“干净的”方向 tuning：shape (n_dirs, n_neurons)
    base_rates = direction_tuning_gaussian(
        theta_stim_deg=directions_deg,
        theta_pref_deg=theta_prefs,
        r_baseline=r_baseline,
        r_max=r_max,
        sigma_deg=tuning_sigma_deg,
    )  # Hz

    # 3) 为每个方向 & 每个 trial 生成 spike train
    n_examples = n_dirs * trials_per_dir
    spikes = np.zeros((n_examples, T_steps, n_neurons), dtype=np.uint8)
    labels = np.zeros(n_examples, dtype=np.int64)

    idx = 0
    for d_idx, theta in enumerate(directions_deg):
        base_rate_dir = base_rates[d_idx]  # (n_neurons,)

        for _ in range(trials_per_dir):
            rates = base_rate_dir.copy()

            # 3.1 trial-by-trial gain noise（multiplicative）
            if gain_sigma > 0.0:
                gain = rng.lognormal(mean=0.0, sigma=gain_sigma)
                rates = rates * gain

            # 3.2 shared additive noise
            if shared_std > 0.0:
                shared = rng.normal(0.0, shared_std)
                rates = rates + shared

            # 3.3 independent additive noise
            if indep_std > 0.0:
                indep = rng.normal(0.0, indep_std, size=n_neurons)
                rates = rates + indep

            rates = np.clip(rates, 0.0, None)

            # 3.4 Poisson spike generation
            spikes_trial = poisson_population_spikes(
                rates_hz=rates,
                T=T,
                dt=dt,
                rng=rng,
            )
            spikes[idx] = spikes_trial
            labels[idx] = d_idx
            idx += 1

    meta = {
        "T": T,
        "dt": dt,
        "r_baseline": r_baseline,
        "r_max_mean": r_max_mean,
        "r_max_std": r_max_std,
        "tuning_sigma_deg": tuning_sigma_deg,
        "jitter_pref_deg": jitter_pref_deg,
        "gain_sigma": gain_sigma,
        "shared_std": shared_std,
        "indep_std": indep_std,
        "n_neurons": n_neurons,
        "trials_per_dir": trials_per_dir,
        "seed": seed,
    }

    out = {
        "spikes": spikes,
        "labels": labels,
        "directions_deg": directions_deg,
        "theta_prefs": theta_prefs,
        "r_max": r_max,
        "meta": meta,
    }
    return out


print("[dataset_builder] loaded. Available names:")
print([name for name in globals().keys() if "generate" in name or "dataset" in name])
