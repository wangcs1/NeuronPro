# analysis/visualization.py
"""
visualization.py
----------------
一些可复用的画图函数：
- plot_confusion_matrix
- plot_accuracy_curves
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    directions_deg: np.ndarray | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
):
    """
    画方向解码的混淆矩阵。

    参数
    ----
    y_true : array-like
        真实类别索引（0..n_dirs-1）
    y_pred : array-like
        预测类别索引
    directions_deg : array-like or None
        每个类别对应方向（deg），用于坐标轴标签。
        如果为 None，则使用类别索引。
    title : str
        图标题
    normalize : bool
        是否按行归一化（即每个真实类别的概率分布）
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
        cm = cm / row_sum

    n_classes = cm.shape[0]
    if directions_deg is None:
        xlabels = np.arange(n_classes)
        ylabels = np.arange(n_classes)
    else:
        xlabels = directions_deg
        ylabels = directions_deg

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, origin="lower", cmap="Blues")
    plt.colorbar(im)

    ticks = np.arange(n_classes)
    plt.xticks(ticks, xlabels, rotation=45)
    plt.yticks(ticks, ylabels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    # 在格子中写数值（可选）
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"
            plt.text(
                j, i, text,
                ha="center", va="center",
                color="black" if val < cm.max() * 0.7 else "white",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


def plot_accuracy_curves(
    x_values: np.ndarray,
    acc_dict: dict[str, np.ndarray],
    x_label: str,
    title: str = "",
    ylim: tuple[float, float] = (0.0, 1.05),
):
    """
    画多条 accuracy 曲线。

    参数
    ----
    x_values : array-like
        自变量取值（如 T, N, sigma）
    acc_dict : dict
        {label: acc_array}，例如：
        {
            "Linear": [0.6, 0.7, 0.8],
            "MLP": [0.65, 0.75, 0.85],
            "SNN": [0.62, 0.8, 0.88],
        }
    x_label : str
        x 轴标签
    title : str
        图标题
    ylim : (float, float)
        y 轴范围
    """
    x_values = np.asarray(x_values, dtype=float)

    plt.figure(figsize=(7, 5))

    markers = ["o", "s", "D", "^", "v", "x"]
    linestyles = ["-", "--", "-.", ":", "-"]
    colors = [None] * 6  # 使用默认颜色循环

    for i, (label, acc) in enumerate(acc_dict.items()):
        acc = np.asarray(acc, dtype=float)
        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        c = colors[i % len(colors)]
        plt.plot(x_values, acc, marker=m, linestyle=ls, label=label)

    plt.xlabel(x_label)
    plt.ylabel("Test accuracy")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.show()
