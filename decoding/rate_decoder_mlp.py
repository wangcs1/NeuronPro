# decoding/rate_decoder_mlp.py
"""
rate_decoder_mlp.py
-------------------
使用简单的 MLP 对基于 rate 的特征进行方向解码。

- 输入特征：每个 trial 的 spike count（沿时间求和）
- 模型：PyTorch MLP
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_rate_features(
    dataset_path: str,
    test_size: float = 0.3,
    random_state: int = 0,
):
    data = np.load(dataset_path, allow_pickle=True)

    spikes = data["spikes"]       # (n_examples, T_steps, n_neurons)
    labels = data["labels"]       # (n_examples,)
    directions_deg = data["directions_deg"]

    X_counts = spikes.sum(axis=1).astype(np.float32)  # (n_examples, n_neurons)
    y = labels.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_counts, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, directions_deg


class RateMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_rate_mlp(
    dataset_path: str,
    test_size: float = 0.1,
    random_state: int = 0,
    batch_size: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
):
    X_train, X_test, y_train, y_test, directions_deg = load_rate_features(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    model = RateMLP(input_dim=input_dim, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def eval_accuracy(loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return accuracy_score(all_labels, all_preds)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            ce_loss = criterion(logits, yb)

            # ---- L2 Regularization ----
            l2_lambda = 1e-4
            l2_loss = 0.0
            for name, param in model.named_parameters():
                if "weight" in name:
                    l2_loss += torch.sum(param ** 2)

            loss = ce_loss + l2_lambda * l2_loss
            # ---------------------------

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            train_acc = eval_accuracy(train_loader)
            test_acc = eval_accuracy(test_loader)
            print(
            f"Epoch {epoch:3d} | "
            f"loss {running_loss/len(train_loader):.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test acc {test_acc:.4f}"
            )

    final_test_acc = eval_accuracy(test_loader)
    print("Final MLP test accuracy:", final_test_acc)

    return model, (final_test_acc, directions_deg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="direction_encoding_static8.npz",
        help="Path to dataset npz file.")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train_rate_mlp(
        dataset_path=args.data,
        n_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
