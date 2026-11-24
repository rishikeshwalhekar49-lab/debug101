import torch
from torch import nn

torch.manual_seed(5)
gen = torch.Generator().manual_seed(17)
n = 2000
points = torch.randn(n, 2, generator=gen) * 1.5
radius = points.norm(dim=1)
labels = ((radius > 1.4) ^ (points[:, 0] > 0)).float()
points = torch.cat([points, points ** 2, points[:, :1] * points[:, 1:]], dim=1)
train_points, test_points = points[:1600], points[1600:]
train_labels, test_labels = labels[:1600], labels[1600:]

model = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
loss_function = nn.BCEWithLogitsLoss()
sgd = torch.optim.SGD(model.parameters(), lr=0.1)

batch_size = 20
steps = 0
while steps < 120:
    shuffle = torch.randperm(len(train_points))
    for k in range(0, len(shuffle), batch_size):
        chosen = shuffle[k:k + batch_size]
        logits = model(train_points[chosen]).squeeze(1)
        loss = loss_function(logits, train_labels[chosen])
        sgd.zero_grad()
        loss.backward()
        sgd.step()
        steps += 1
        if steps % 30 == 0:
            with torch.no_grad():
                test_loss = loss_function(model(test_points).squeeze(1), test_labels).item()
            print(steps, loss.item(), test_loss)
        if steps >= 120:
            break
