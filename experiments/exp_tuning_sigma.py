
"""
exp_tuning_sigma.py
-------------------
实验：tuning 宽度 sigma_deg 对方向解码性能的影响。
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from encoding.dataset_builder import generate_direction_encoding_dataset
from decoding.linear_decoder import train_and_eval_linear
from decoding.rate_decoder_mlp import train_rate_mlp
from decoding.snn_decoder import train_snn_decoder


def run_tuning_sigma_experiment(
    sigma_list_deg = (20.0, 30.0, 45.0, 60.0, 80.0),
    base_seed: int = 200,
    save_datasets: bool = False,
):
    directions = np.arange(0, 360, 45)  # 8 个方向
    n_neurons = 40
    trials_per_dir = 100
    T = 400.0   # ms

    acc_lin = []
    acc_mlp = []
    acc_snn = []

    for i, sigma in enumerate(sigma_list_deg):
        print("\n" + "=" * 60)
        print(f"[Tuning Sigma Experiment] sigma = {sigma} deg")
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
            tuning_sigma_deg=sigma,
            jitter_pref_deg=7.0,
            gain_sigma=0.25,
            shared_std=3.0,
            indep_std=2.0,
            seed=base_seed + i,
        )

        spikes = dataset["spikes"]
        labels = dataset["labels"]
        directions_deg = dataset["directions_deg"]

        tmp_path = f"tmp_tuning_sigma_{int(sigma)}deg.npz"
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
    sigma_arr = np.array(sigma_list_deg, dtype=float)

    plt.plot(sigma_arr, acc_lin, marker="o", linestyle="-",  label="Linear (LogReg)")
    plt.plot(sigma_arr, acc_mlp, marker="s", linestyle="--", label="MLP (rate)")
    plt.plot(sigma_arr, acc_snn, marker="D", linestyle="-.", label="SNN (spike train)")

    plt.xlabel("Tuning width σ (deg)")
    plt.ylabel("Test accuracy")
    plt.title("Effect of Tuning Width on Direction Decoding")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "sigma_list": sigma_arr,
        "acc_lin": np.array(acc_lin),
        "acc_mlp": np.array(acc_mlp),
        "acc_snn": np.array(acc_snn),
    }


if __name__ == "__main__":
    run_tuning_sigma_experiment()
