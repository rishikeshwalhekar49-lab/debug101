import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(2)
g = torch.Generator().manual_seed(11)
inputs = torch.randn(1200, 12, generator=g)
weights = torch.randn(12, 1, generator=g)
targets = torch.sin(inputs @ weights) + 0.5 * (inputs[:, :1] ** 2) + 0.05 * torch.randn(1200, 1, generator=g)
train_x, val_x = inputs[:1000], inputs[1000:]
train_y, val_y = targets[:1000], targets[1000:]


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x):
        return x + self.ff(self.norm(x))


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(12, 32)
        self.blocks = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


model = Regressor()
objective = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=50, shuffle=True)
total_steps = 5 * len(loader)
schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))

for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = objective(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        schedule.step()
    with torch.no_grad():
        val_loss = objective(model(val_x), val_y).item()
    print(f"epoch {epoch}: {loss.item():.4f} / {val_loss:.4f}")
