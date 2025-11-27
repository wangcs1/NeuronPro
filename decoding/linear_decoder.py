# decoding/linear_decoder.py
"""
linear_decoder.py
-----------------
使用 Logistic Regression 对方向进行解码（线性基线）。

- 输入特征：每个 trial 的 spike count（沿时间求和）
- 模型：sklearn.linear_model.LogisticRegression
"""

from __future__ import annotations
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_rate_features(
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 0,
):
    """从 npz 数据集中构造 rate 特征并做 train/test 划分。"""
    data = np.load(dataset_path, allow_pickle=True)

    spikes = data["spikes"]  # (n_examples, T_steps, n_neurons)
    labels = data["labels"]  # (n_examples,)
    directions_deg = data["directions_deg"]

    # rate 特征：每个 neuron 在整个 trial 内的 spike 总数
    X_counts = spikes.sum(axis=1).astype(np.float32)  # (n_examples, n_neurons)
    y = labels.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_counts, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, directions_deg


def train_and_eval_linear(
    dataset_path: str,
    C: float = 1.0,
    max_iter: int = 1000,
    test_size: float = 0.2,
    random_state: int = 0,
):
    X_train, X_test, y_train, y_test, directions_deg = load_rate_features(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
    )

    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
    print("y_train distribution:",
          {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})
    print("y_test  distribution:",
          {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))})

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Logistic Regression train acc: {train_acc:.4f}")
    print(f"Logistic Regression test  acc: {test_acc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, origin="lower", cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(directions_deg))
    plt.xticks(ticks, directions_deg, rotation=45)
    plt.yticks(ticks, directions_deg)
    plt.xlabel("Predicted direction (deg)")
    plt.ylabel("True direction (deg)")
    plt.title("Linear decoder (LogReg) confusion matrix")
    plt.tight_layout()
    plt.show()

    return clf, (train_acc, test_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="direction_encoding_static8.npz",
        help="Path to dataset npz file.")
    args = parser.parse_args()

    train_and_eval_linear(args.data)


if __name__ == "__main__":
    main()
