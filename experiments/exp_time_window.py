
"""
exp_time_window.py
------------------
实验：时间窗口长度 T 对方向解码性能的影响。
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
    directions = np.arange(0, 360, 45) 
    n_neurons = 40
    trials_per_dir = 100

    acc_lin = []
    acc_mlp = []
    acc_snn = []

    for i, T in enumerate(T_list_ms):
        print("\n" + "=" * 60)
        print(f"[Time Window Experiment] T = {T} ms")
        print("=" * 60)
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
        tmp_path = f"tmp_time_T_{int(T)}ms.npz"
        np.savez(
            tmp_path,
            spikes=spikes,
            labels=labels,
            directions_deg=directions_deg,
        )
        print(f"Saved temporary dataset: {tmp_path} | spikes shape: {spikes.shape}")

        _, (train_acc_lin, test_acc_lin) = train_and_eval_linear(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
        )
        acc_lin.append(test_acc_lin)

        _, (test_acc_mlp, _) = train_rate_mlp(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
            n_epochs=40, 
        )
        acc_mlp.append(test_acc_mlp)

        _, (test_acc_snn, _) = train_snn_decoder(
            dataset_path=tmp_path,
            test_size=0.2,
            random_state=0,
            n_epochs=40,
        )
        acc_snn.append(test_acc_snn)
        if not save_datasets and os.path.exists(tmp_path):
            os.remove(tmp_path)

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
