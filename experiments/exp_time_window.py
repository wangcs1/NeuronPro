# experiments/exp_time_window.py
"""
exp_time_window.py
------------------
实验：时间窗口长度 T 对方向解码性能的影响。

固定：
- 方向数：8 个（0..315，每 45°）
- 神经元数：40
- tuning 宽度 / 噪声：适中
- 每个方向 trial 数：100

变量：
- T ∈ {50, 100, 200, 400, 800} ms

对每个 T：
- 用 Poisson + tuning 生成 spikes
- 分别训练：
    - 线性 Logistic 回归（基于 rate）
    - MLP（基于 rate）
    - SNN（基于 spike train）
- 记录 test acc 并画成曲线。
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from encoding.dataset_builder import generate_direction_encoding_dataset
from decoding.linear_decoder import train_and_eval_linear
from decoding.rate_decoder_mlp import train_rate_mlp
from decoding.snn_decoder import train_snn_decoder


def run_time_window_experiment(
    T_list_ms = (50.0, 100.0, 200.0, 400.0, 800.0),
    base_seed: int = 0,
    save_datasets: bool = False,
):
    directions = np.arange(0, 360, 45)  # 8 个方向
    n_neurons = 40
    trials_per_dir = 100

    # 用列表记录不同 T 下三种解码器的 acc
    acc_lin = []
    acc_mlp = []
    acc_snn = []

    for i, T in enumerate(T_list_ms):
        print("\n" + "=" * 60)
        print(f"[Time Window Experiment] T = {T} ms")
        print("=" * 60)

        # 1) 生成数据
        dataset = generate_direction_encoding_dataset(
            directions_deg=directions,
            n_neurons=n_neurons,
            trials_per_dir=trials_per_dir,
            T=T,
            dt=1.0,
            r_baseline=8.0,
            r_max_mean=25.0,
            r_max_std=6.0,
            tuning_sigma_deg=50.0,
            jitter_pref_deg=7.0,
            gain_sigma=0.25,
            shared_std=3.0,
            indep_std=2.0,
            seed=base_seed + i,
        )

        spikes = dataset["spikes"]
        labels = dataset["labels"]
        directions_deg = dataset["directions_deg"]

        # 2) 保存成临时 npz，方便复用已有 decoding 代码
        tmp_path = f"tmp_time_T_{int(T)}ms.npz"
        np.savez(
            tmp_path,
            spikes=spikes,
            labels=labels,
            directions_deg=directions_deg,
        )
        print(f"Saved temporary dataset: {tmp_path} | spikes shape: {spikes.shape}")

        # 3) 线性 decoder
        _, (train_acc_lin, test_acc_lin) = train_and_eval_linear(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
        )
        acc_lin.append(test_acc_lin)

        # 4) MLP decoder（基于 rate）
        _, (test_acc_mlp, _) = train_rate_mlp(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
            n_epochs=40,  # sweep 时 epoch 可以略少
        )
        acc_mlp.append(test_acc_mlp)

        # 5) SNN decoder（基于 spike train）
        _, (test_acc_snn, _) = train_snn_decoder(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
            n_epochs=40,
        )
        acc_snn.append(test_acc_snn)

        # 6) 如果不需要保留数据集，可以删除临时文件
        if not save_datasets and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 7) 画图
    plt.figure(figsize=(7, 5))
    T_arr = np.array(T_list_ms, dtype=float)

    plt.plot(T_arr, acc_lin,  marker="o", linestyle="-",  label="Linear (LogReg)")
    plt.plot(T_arr, acc_mlp,  marker="s", linestyle="--", label="MLP (rate)")
    plt.plot(T_arr, acc_snn,  marker="D", linestyle="-.", label="SNN (spike train)")

    plt.xlabel("Time window T (ms)")
    plt.ylabel("Test accuracy")
    plt.title("Effect of Time Window on Direction Decoding")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "T_list": T_arr,
        "acc_lin": np.array(acc_lin),
        "acc_mlp": np.array(acc_mlp),
        "acc_snn": np.array(acc_snn),
    }


if __name__ == "__main__":
    run_time_window_experiment()
