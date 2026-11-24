import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)


def make_data(n, d=20, classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, d, generator=torch.Generator().manual_seed(7)) * 1.5
    labels = torch.randint(0, classes, (n,), generator=g)
    features = centers[labels] + torch.randn(n, d, generator=g)
    return features, labels


x_tr, y_tr = make_data(1024, seed=1)
x_va, y_va = make_data(256, seed=2)

net = nn.Sequential(nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)


def train_epoch():
    net.train()
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate():
    net.eval()
    logits = net(x_va)
    return loss_fn(logits, y_va).item(), (logits.argmax(1) == y_va).float().mean().item()


for epoch in range(6):
    train_loss = train_epoch()
    val_loss, acc = evaluate()
    print(f"epoch {epoch}: train {train_loss:.3f} val {val_loss:.3f} acc {acc:.3f}")
