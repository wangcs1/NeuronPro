# decoding/snn_decoder.py
"""
snn_decoder.py
--------------
使用 snntorch 基于 spike train 进行方向解码。

- 输入特征：完整 spike train (batch, T_steps, n_neurons)
- 模型:fc → LIF → fc,时间维上迭代,time-sum 作为读出
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_spike_dataset(
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 0,
):
    data = np.load(dataset_path, allow_pickle=True)

    spikes = data["spikes"].astype(np.float32)     # (n_examples, T_steps, n_neurons)
    labels = data["labels"].astype(np.int64)       # (n_examples,)
    directions_deg = data["directions_deg"]

    X_train, X_test, y_train, y_test = train_test_split(
        spikes, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, directions_deg


spike_grad = surrogate.fast_sigmoid(slope=25.0)


class SNNDirectionDecoder(nn.Module):
    """fc → LIF → fc 的简单 SNN 解码器。"""

    def __init__(self, input_size: int, hidden_size: int, n_classes: int, beta: float = 0.9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, T_steps, input_size)
        返回: (batch_size, n_classes) 的 logits
        """
        batch_size, T_steps, input_size = x.shape
        device = x.device

        mem1 = self.lif1.init_leaky()
        out_rec = []

        for t in range(T_steps):
            x_t = x[:, t, :]          # (batch, input_size)
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)     # (batch, n_classes)
            out_rec.append(cur2.unsqueeze(1))

        out_rec = torch.cat(out_rec, dim=1)  # (batch, T_steps, n_classes)
        out_sum = out_rec.sum(dim=1)         # time-sum readout
        return out_sum


def eval_accuracy(model: nn.Module, loader, device: torch.device) -> float:
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


def train_snn_decoder(
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 0,
    batch_size: int = 32,
    n_epochs: int = 50,
    lr: float = 1e-3,
    hidden_size: int = 128,
    beta: float = 0.9,
):
    X_train, X_test, y_train, y_test, directions_deg = load_spike_dataset(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]
    n_classes = len(np.unique(y_train.numpy()))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SNNDirectionDecoder(
        input_size=input_size,
        hidden_size=hidden_size,
        n_classes=n_classes,
        beta=beta,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            train_acc = eval_accuracy(model, train_loader, device)
            test_acc = eval_accuracy(model, test_loader, device)
            print(
                f"Epoch {epoch:3d} | "
                f"loss {running_loss/len(train_loader):.4f} | "
                f"train acc {train_acc:.4f} | "
                f"test acc {test_acc:.4f}"
            )

    final_test_acc = eval_accuracy(model, test_loader, device)
    print("Final SNN test accuracy:", final_test_acc)

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

    train_snn_decoder(
        dataset_path=args.data,
        n_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
