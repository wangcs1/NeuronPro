# convert_v4_2_to_npz.py
"""
把 CRCNS v4-2 数据集（HenryKohn2022_*.mat）里的 V4 population
转成你的项目可用的 .npz 格式。

输出两类文件：
1. 每个 session 各自一个 npz：
    <base>_sessionX_v4.npz
    内含：
        spikes      : (n_trials, T_steps, n_neurons)
                      这里的 spikes 是 [0,1] 的连续值，表示每个 time bin 内的发放概率 / 归一化 rate
        spikecounts : (n_trials, n_neurons)   # 原始 250 ms 总 spike 数
        labels      : (n_trials,)
        directions_deg : (n_dirs,)
        meta        : dict（包含 session = X 等）

2. 合并所有 session 的一个“大 npz”：
    <base>_all_sessions_v4.npz
    内含：
        spikes      : (N_all_trials, T_steps, n_neurons)  # analog [0,1]
        spikecounts : (N_all_trials, n_neurons)
        labels      : (N_all_trials,)
        directions_deg : (n_dirs,)
        session_idx : (N_all_trials,)  # 记录每个 trial 来自哪个 session
        meta        : dict（包含各 session n_trials 列表等）

注意：
- 真实数据只给出了 0.25 s 内的总 spike 数，没有时间点。
  这里我们假设在 250 ms 内是平稳 Poisson 发放，用
      p_bin = rate_hz * (dt/1000)
  作为该 time bin 内的发放概率，并把这个概率在时间上复制，
  得到一个 analog spike train（而不是随机 0/1 事件）。
- 这样 Linear/MLP 仍然可以用 spikecounts 做 rate 解码，
  SNN 则可以吃 (T_steps, n_neurons) 的 analog spike 序列。
"""

import os
import numpy as np
from scipy.io import loadmat

# ----------------- 一些全局配置 -----------------

# 每个刺激 epoch 的时长（来自 data_description：0.25 s）
T_EPOCH_MS = 250.0
# 我们用于“展开”时间的步长
DT_MS = 5.0    # 5 ms → T_steps = 50

# target orientation 对应的参数名提示（用于自动查找）
# 如果自动匹配失败，你可以把这里改成精确的名字，比如 'target_orientation'
ORI_PARAM_NAME_HINT = "target"   # 包含 'target' 且 'ori'/'orient'


def poisson_population_spikes_analog(rates_hz, T, dt):
    """
    方案 A：不采样 0/1 spike，而是直接用 Poisson 概率作为 analog 强度。

    rates_hz : (n_neurons,)
        对应 250 ms 窗口内的平均 firing rate (Hz)
    T : float
        窗口时长，单位 ms
    dt : float
        时间步长，单位 ms

    返回：
        spikes_analog : (T_steps, n_neurons) float32
        每个元素在 [0,1]，表示该 time bin 内的发放概率 / 归一化 rate。
        在时间维上是把同一个 p 向量复制 T_steps 次，因此没有时间结构，
        但对 SNN 来说是一个「持续输入电流」。
    """
    rates_hz = np.asarray(rates_hz, dtype=float)
    T_steps = int(T / dt)
    # rate * Δt(s) ≈ 该 bin 内期望 spike 数 ≈ 发放概率
    p = rates_hz * (dt / 1000.0)
    p = np.clip(p, 0.0, 1.0)
    spikes_analog = np.tile(p[None, :], (T_steps, 1))
    return spikes_analog.astype(np.float32)


def find_orientation_param_index(param_labels, name_hint=None):
    """
    param_labels: list[str]，来自 MATLAB 的 parameter_labels
    自动找出“target orientation”对应的参数下标。

    策略：
    1. 如果 name_hint 不为空，优先找包含该子串的字段
    2. 否则找包含 'target' 且 ('ori' 或 'orient') 的字段
    """
    labels_lower = [str(l).strip().lower() for l in param_labels]

    # 1) 先用 hint
    if name_hint is not None:
        hint = name_hint.lower()
        for i, lab in enumerate(labels_lower):
            if hint in lab:
                return i

    # 2) 否则用 heuristic
    candidates = []
    for i, lab in enumerate(labels_lower):
        if "target" in lab and ("ori" in lab or "orient" in lab):
            candidates.append(i)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # 有多个就先返回第一个，并打印提示
        print("[WARN] 找到多个可能的 target orientation 参数，默认使用第一个：")
        for idx in candidates:
            print(f"  idx={idx}, label='{param_labels[idx]}'")
        return candidates[0]
    else:
        raise RuntimeError(
            f"无法在 parameter_labels={param_labels} 中自动找到 target orientation 参数，"
            f"请修改 ORI_PARAM_NAME_HINT 或手动指定 index。"
        )


def convert_single_mat(
    mat_path,
    out_dir,
    area_name="v4population",
    ori_param_name_hint=ORI_PARAM_NAME_HINT,
    dt_ms=DT_MS,
    T_epoch_ms=T_EPOCH_MS,
    seed=0,
):
    """
    处理一份 HenryKohn2022_*.mat：
    - 仍然为每个 session 输出一个小 npz
    - 同时把所有 session 合成一个大的 npz

    关键点：
    - 先做一遍预扫描，找到所有 session 的最小 neuron 数 n_common，
      然后对每个 session 只保留前 n_common 个 neuron，以便在 neuron 维度上 concat。
    - 使用方案 A：spikes 是 analog 概率值 [0,1]，不再是 0/1 采样。
    """
    print(f"\n=== Converting {mat_path} (area={area_name}) ===")
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if area_name not in mat:
        raise KeyError(f"{area_name} not found in {mat.keys()}")

    population = mat[area_name]
    if not isinstance(population, np.ndarray):
        population = np.array([population])
    n_sessions = population.size
    print(f"Found {n_sessions} sessions.")

    # 读取 parameter labels（用第一 session）
    param_labels_raw = np.ravel(population[0].parameter_labels)
    param_labels = [str(l) for l in param_labels_raw]
    print("parameter_labels:", param_labels)

    ori_param_idx = find_orientation_param_index(param_labels, ori_param_name_hint)
    print(f"Using parameter index {ori_param_idx} for target orientation.")

    # ========= 第 1 轮扫描：只看各 session 的 n_neurons 和 directions =========
    directions_global = None
    n_neurons_list = []
    per_session_n_trials = []

    for s_idx in range(n_sessions):
        sess = population[s_idx]
        spikecounts_raw = np.asarray(sess.spikecounts, dtype=float)  # (n_neurons_s, n_trials_s)
        n_neurons_s, n_trials_s = spikecounts_raw.shape
        n_neurons_list.append(n_neurons_s)
        per_session_n_trials.append(int(n_trials_s))

        params = np.asarray(sess.parameters, dtype=float)            # (n_params, n_trials_s)
        ori_deg_s = params[ori_param_idx]
        directions_deg = np.sort(np.unique(ori_deg_s))

        if directions_global is None:
            directions_global = directions_deg
        else:
            if not np.array_equal(directions_global, directions_deg):
                raise ValueError(
                    f"Session {s_idx} 的 directions_deg={directions_deg} "
                    f"与之前 session 不一致={directions_global}，"
                    "目前脚本假设所有 session 的方向集合一致。"
                )

    n_common_neurons = min(n_neurons_list)
    print("Neuron numbers per session:", n_neurons_list)
    print("Using common neuron dimension n_common_neurons =", n_common_neurons)

    rng = np.random.default_rng(seed)  # 这里 rng 只是为了保持接口一致，方案 A 中实际上不会用到它
    os.makedirs(out_dir, exist_ok=True)

    # ========= 第 2 轮：真正生成每个 session 的 spikes，并累计到 big npz =========
    all_spikes = []
    all_spikecounts = []
    all_labels = []
    all_session_idx = []

    for s_idx in range(n_sessions):
        print(f"\n--- Processing session {s_idx} ---")
        sess = population[s_idx]

        spikecounts_raw = np.asarray(sess.spikecounts, dtype=float)  # (n_neurons_s, n_trials_s)
        # 只保留前 n_common_neurons 个 neuron
        spikecounts_raw = spikecounts_raw[:n_common_neurons, :]      # (n_common_neurons, n_trials_s)
        spikecounts_s = spikecounts_raw.T                             # → (n_trials_s, n_common_neurons)
        n_trials_s = spikecounts_s.shape[0]
        print(f"  n_trials={n_trials_s}, n_neurons(after cut)={n_common_neurons}")

        params = np.asarray(sess.parameters, dtype=float)           # (n_params, n_trials_s)
        ori_deg_s = params[ori_param_idx]
        directions_deg = directions_global  # 已经在第一轮确定

        # build label
        labels_s = np.zeros_like(ori_deg_s, dtype=np.int64)
        for k, theta in enumerate(directions_deg):
            labels_s[ori_deg_s == theta] = k

        # generate analog "spike" trains in [0,1]
        T_steps = int(T_epoch_ms / dt_ms)
        spikes_s = np.zeros((n_trials_s, T_steps, n_common_neurons), dtype=np.float32)
        for t in range(n_trials_s):
            rates_hz = spikecounts_s[t] / (T_epoch_ms / 1000.0)            # (n_neurons,)
            spikes_s[t] = poisson_population_spikes_analog(rates_hz, T_epoch_ms, dt_ms)

        # 保存“小 npz”： <base>_sessionX_v4.npz
        base = os.path.basename(mat_path).replace(".mat", "")
        out_npz_sess = os.path.join(out_dir, f"{base}_session{s_idx}_v4.npz")

        meta_s = {
            "source": "CRCNS_v4-2",
            "mat_file": os.path.basename(mat_path),
            "area_name": area_name,
            "session": s_idx,
            "n_neurons": int(n_common_neurons),
            "n_trials": int(n_trials_s),
            "directions_deg": directions_deg.tolist(),
            "T_epoch_ms": T_epoch_ms,
            "dt_ms": dt_ms,
            "spikes_encoding": "analog_probability",  # 明确标记一下
        }

        np.savez(
            out_npz_sess,
            spikes=spikes_s,
            spikecounts=spikecounts_s,
            labels=labels_s,
            directions_deg=directions_deg,
            meta=np.array([meta_s], dtype=object)
        )
        print("Saved session npz:", out_npz_sess)

        # 累积到 big npz
        all_spikes.append(spikes_s)
        all_spikecounts.append(spikecounts_s)
        all_labels.append(labels_s)
        all_session_idx.append(np.full(n_trials_s, s_idx, dtype=np.int64))

    # ========= 合成“大 npz” =========
    print("\n=== Merging all sessions into one big npz ===")
    spikes_big = np.concatenate(all_spikes, axis=0)         # (N_all_trials, T_steps, n_common_neurons)
    spikecounts_big = np.concatenate(all_spikecounts, axis=0)
    labels_big = np.concatenate(all_labels, axis=0)
    session_idx_big = np.concatenate(all_session_idx, axis=0)

    print("  spikes_big shape     :", spikes_big.shape)
    print("  spikecounts_big shape:", spikecounts_big.shape)
    print("  labels_big shape     :", labels_big.shape)
    print("  session_idx_big shape:", session_idx_big.shape)
    print("  directions_deg       :", directions_global)

    base = os.path.basename(mat_path).replace(".mat", "")
    out_npz_big = os.path.join(out_dir, f"{base}_all_sessions_v4.npz")

    meta_big = {
        "source": "CRCNS_v4-2",
        "mat_file": os.path.basename(mat_path),
        "area_name": area_name,
        "n_sessions": int(n_sessions),
        "per_session_n_trials": per_session_n_trials,
        "n_neurons": int(n_common_neurons),
        "N_all_trials": int(spikecounts_big.shape[0]),
        "directions_deg": directions_global.tolist(),
        "T_epoch_ms": T_epoch_ms,
        "dt_ms": dt_ms,
        "spikes_encoding": "analog_probability",
    }

    np.savez(
        out_npz_big,
        spikes=spikes_big,
        spikecounts=spikecounts_big,
        labels=labels_big,
        directions_deg=directions_global,
        session_idx=session_idx_big,
        meta=np.array([meta_big], dtype=object),
    )
    print("Saved BIG npz:", out_npz_big)


if __name__ == "__main__":
    # 你可以根据自己的路径修改这里
    base_dir = r"FinPro\mat_data"
    out_dir = r"FinPro\npz_data"

    # 1) distractor orientation 实验
    convert_single_mat(
        mat_path=os.path.join(base_dir, "HenryKohn2022_distractororientation.mat"),
        out_dir=out_dir,
        area_name="v4population",
    )

    # 2) distractor separation 实验（如果你也要处理的话）
    convert_single_mat(
        mat_path=os.path.join(base_dir, "HenryKohn2022_distractorseparation.mat"),
        out_dir=out_dir,
        area_name="v4population",
    )
