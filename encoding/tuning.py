# encoding/tuning.py
"""
tuning.py
---------
定义视觉方向选择性神经元的 tuning curve。
核心功能：
- 生成一组神经元的 preferred direction
- 根据 tuning 曲线把刺激方向映射为 firing rate(Hz)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def circ_dist_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算圆周上的角度差（单位：度），结果范围在 [-180, 180)。

    参数
    ----
    a : array-like
        角度（度），可以是标量或数组
    b : array-like
        角度（度），可以是标量或数组，与 a 广播兼容

    返回
    ----
    d : ndarray
        a - b 的圆周差，范围 [-180, 180)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def generate_preferred_directions(
    n_neurons: int,
    jitter_deg: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    为一组方向选择性神经元生成 preferred direction(度)。

    思路：
    - 先在 [0, 360) 上均匀分布 n_neurons 个角度
    - 再加上一点高斯随机抖动（可选）

    参数
    ----
    n_neurons : int
        神经元数量
    jitter_deg : float, default 0.0
        对每个神经元 preferred direction 添加的高斯随机抖动标准差(度)
    seed : int | None
        随机种子，方便复现实验

    返回
    ----
    theta_prefs : ndarray, shape (n_neurons,)
        每个神经元的 preferred direction(度)
    """
    rng = np.random.default_rng(seed)
    theta_prefs = np.linspace(0.0, 360.0, n_neurons, endpoint=False)
    if jitter_deg > 0.0:
        theta_prefs = theta_prefs + rng.normal(0.0, jitter_deg, size=n_neurons)
    # wrap 回 [0, 360)
    theta_prefs = np.mod(theta_prefs, 360.0)
    return theta_prefs


def sample_r_max(
    n_neurons: int,
    r_max_mean: float = 20.0,
    r_max_std: float = 5.0,
    min_rate: float = 1.0,
    max_rate: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    为每个神经元采样一个最大调制幅度 r_max_i(Hz)。

    参数
    ----
    n_neurons : int
        神经元数量
    r_max_mean : float
        调制幅度均值(Hz)
    r_max_std : float
        调制幅度标准差(Hz)
    min_rate : float
        最小允许的 r_max(裁剪用)
    max_rate : float | None
        最大允许的 r_max,如果为 None 则不裁剪上界
    seed : int | None
        随机种子

    返回
    ----
    r_max : ndarray, shape (n_neurons,)
        每个神经元的最大调制幅度(Hz)
    """
    rng = np.random.default_rng(seed)
    r_max = rng.normal(r_max_mean, r_max_std, size=n_neurons)
    if max_rate is not None:
        r_max = np.clip(r_max, min_rate, max_rate)
    else:
        r_max = np.clip(r_max, min_rate, None)
    return r_max


def direction_tuning_gaussian(
    theta_stim_deg: np.ndarray,
    theta_pref_deg: np.ndarray,
    r_baseline: float,
    r_max: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    """
    高斯型（在圆周上展开的）方向 tuning curve:

        f_i(θ) = r_baseline + r_max_i * exp( - Δθ_i^2 / (2 σ^2) )

    其中 Δθ_i 是 θ 与 θ_pref_i 的圆周角度差（考虑 0/360 wrap)。

    参数
    ----
    theta_stim_deg : array-like, shape (n_dirs,)
        刺激方向集合（度）
    theta_pref_deg : array-like, shape (n_neurons,)
        每个神经元的 preferred direction（度）
    r_baseline : float
        baseline firing rate(Hz)
    r_max : array-like, shape (n_neurons,)
        每个神经元的最大调制幅度（Hz）
    sigma_deg : float
        tuning 曲线的宽度参数（标准差，度）

    返回
    ----
    rates : ndarray, shape (n_dirs, n_neurons)
        对于每个刺激方向和每个神经元的 firing rate（Hz）
    """
    theta_stim_deg = np.asarray(theta_stim_deg)  # (n_dirs,)
    theta_pref_deg = np.asarray(theta_pref_deg)  # (n_neurons,)
    r_max = np.asarray(r_max)                    # (n_neurons,)

    # 广播成 (n_dirs, n_neurons)
    # stim[:, None] - pref[None, :] → (n_dirs, n_neurons)
    dtheta = circ_dist_deg(theta_stim_deg[:, None], theta_pref_deg[None, :])
    rates = r_baseline + r_max[None, :] * np.exp(
        - (dtheta ** 2) / (2.0 * sigma_deg ** 2)
    )

    # 保证非负（理论上不会为负，但考虑数值安全）
    rates = np.clip(rates, 0.0, None)
    return rates
