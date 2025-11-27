# make_dataset_static8.py （可以放在根目录或 scripts/ 下）
import numpy as np
import os

from encoding.dataset_builder import generate_direction_encoding_dataset

def main():
    # 8 个静态方向
    directions = np.arange(0, 360, 45)  # 0,45,...,315

    dataset = generate_direction_encoding_dataset(
        directions_deg=directions,
        n_neurons=40,
        trials_per_dir=100,
        T=400.0,
        dt=1.0,
        r_baseline=8.0,
        r_max_mean=25.0,
        r_max_std=6.0,
        tuning_sigma_deg=50.0,
        jitter_pref_deg=7.0,
        gain_sigma=0.25,
        shared_std=3.0,
        indep_std=2.0,
        seed=0,
    )

    np.savez(
        "direction_encoding_static8.npz",
        spikes=dataset["spikes"],
        labels=dataset["labels"],
        directions_deg=dataset["directions_deg"],
        theta_prefs=dataset["theta_prefs"],
        r_max=dataset["r_max"],
        meta=np.array([dataset["meta"]], dtype=object),  # meta 用 object 存
    )
    print("Saved dataset to direction_encoding_static8.npz")
    print("spikes shape:", dataset["spikes"].shape)

if __name__ == "__main__":
    main()
