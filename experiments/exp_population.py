"""
exp_population.py
-----------------
实验：群体神经元数量 N 对方向解码性能的影响。

"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from encoding.dataset_builder import generate_direction_encoding_dataset
from decoding.linear_decoder import train_and_eval_linear
from decoding.rate_decoder_mlp import train_rate_mlp
from decoding.snn_decoder import train_snn_decoder


def run_population_experiment(
    N_list = (10, 20, 40, 80),
    base_seed: int = 100,
    save_datasets: bool = False,
):
    directions = np.arange(0, 360, 45)  # 8 个方向
    trials_per_dir = 100
    T = 400.0   # ms

    acc_lin = []
    acc_mlp = []
    acc_snn = []

    for i, N in enumerate(N_list):
        print("\n" + "=" * 60)
        print(f"[Population Experiment] N_neurons = {N}")
        print("=" * 60)

        dataset = generate_direction_encoding_dataset(
            directions_deg=directions,
            n_neurons=N,
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

        tmp_path = f"tmp_population_N_{N}.npz"
        np.savez(
            tmp_path,
            spikes=spikes,
            labels=labels,
            directions_deg=directions_deg,
        )
        print(f"Saved temporary dataset: {tmp_path} | spikes shape: {spikes.shape}")

        # Linear
        _, (train_acc_lin, test_acc_lin) = train_and_eval_linear(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
        )
        acc_lin.append(test_acc_lin)

        # MLP
        _, (test_acc_mlp, _) = train_rate_mlp(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
            n_epochs=40,
        )
        acc_mlp.append(test_acc_mlp)

        # SNN
        _, (test_acc_snn, _) = train_snn_decoder(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
            n_epochs=40,
        )
        acc_snn.append(test_acc_snn)

        if not save_datasets and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 画图
    plt.figure(figsize=(7, 5))
    N_arr = np.array(N_list, dtype=int)

    plt.plot(N_arr, acc_lin, marker="o", linestyle="-",  label="Linear (LogReg)")
    plt.plot(N_arr, acc_mlp, marker="s", linestyle="--", label="MLP (rate)")
    plt.plot(N_arr, acc_snn, marker="D", linestyle="-.", label="SNN (spike train)")

    plt.xlabel("Number of neurons (N)")
    plt.ylabel("Test accuracy")
    plt.title("Effect of Population Size on Direction Decoding")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.xticks(N_arr)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "N_list": N_arr,
        "acc_lin": np.array(acc_lin),
        "acc_mlp": np.array(acc_mlp),
        "acc_snn": np.array(acc_snn),
    }


if __name__ == "__main__":
    run_population_experiment()
