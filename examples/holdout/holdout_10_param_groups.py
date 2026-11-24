import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(9)
rng = torch.Generator().manual_seed(41)
features = torch.randn(1400, 24, generator=rng)
projection = torch.randn(24, 5, generator=rng)
classes = (torch.relu(features) @ projection).argmax(1)
dataset = TensorDataset(features[:1200], classes[:1200])
holdout = (features[1200:], classes[1200:])


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(24, 96), nn.ReLU(), nn.Dropout(0.2), nn.Linear(96, 48), nn.ReLU())
        self.classifier = nn.Linear(48, 5)

    def forward(self, x):
        return self.classifier(self.backbone(x))


network = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": network.backbone.parameters(), "lr": 1e-3},
    {"params": network.classifier.parameters(), "lr": 3e-3},
])
batches = DataLoader(dataset, batch_size=40, shuffle=True)

for epoch in range(4):
    network.train()
    for inputs, labels in batches:
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        val_loss = criterion(network(holdout[0]), holdout[1]).item()
    print(f"epoch={epoch} loss={loss.item():.4f} val={val_loss:.4f}")
