# encoding/poisson_spike.py
"""
poisson_spike.py
----------------
根据给定的 firing rate（Hz）生成 Poisson spike train。

使用的是离散时间近似：
  p_spike = rate_hz * (dt / 1000)
在每个时间步进行一次 Bernoulli 抽样。
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def poisson_spike_train(
    rate_hz: float,
    T: float,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    为单个神经元生成 Poisson spike train。

    参数
    ----
    rate_hz : float
        firing rate（Hz）
    T : float
        trial 时长（ms）
    dt : float
        时间步长（ms）
    rng : np.random.Generator | None
        随机数生成器

    返回
    ----
    spikes : ndarray, shape (T_steps,)
        0/1 序列，1 表示该时间步有 spike。
    """
    if rng is None:
        rng = np.random.default_rng()
    T_steps = int(T / dt)

    # dt(ms) → dt(s)
    p_spike = rate_hz * (dt / 1000.0)
    # 防止数值不合法
    p_spike = np.clip(p_spike, 0.0, 1.0)

    # 每个时间步一个 Bernoulli
    spikes = rng.random(size=T_steps) < p_spike
    return spikes.astype(np.uint8)


def poisson_population_spikes(
    rates_hz: np.ndarray,
    T: float,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    为一群神经元生成 Poisson spike train。

    参数
    ----
    rates_hz : ndarray, shape (n_neurons,)
        每个神经元的 firing rate（Hz）
    T : float
        trial 时长（ms）
    dt : float
        时间步长（ms）
    rng : np.random.Generator | None
        随机数生成器

    返回
    ----
    spikes : ndarray, shape (T_steps, n_neurons)
        0/1 spike 矩阵。
    """
    if rng is None:
        rng = np.random.default_rng()
    rates_hz = np.asarray(rates_hz)
    T_steps = int(T / dt)
    n_neurons = rates_hz.shape[0]

    # (n_neurons,) → (1, n_neurons) 再广播到 (T_steps, n_neurons)
    p_spike = rates_hz[None, :] * (dt / 1000.0)
    p_spike = np.clip(p_spike, 0.0, 1.0)

    # 生成一个 [0,1) 随机矩阵
    rand_mat = rng.random(size=(T_steps, n_neurons))
    spikes = rand_mat < p_spike
    return spikes.astype(np.uint8)
