import numpy as np
import torch
from torch import nn
from torch.utils import data

SAMPLE_DATA = "/content/sample_data"

train = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
test = np.loadtxt(f"{SAMPLE_DATA}/mnist_test.csv", delimiter=",", dtype=np.float32, max_rows=1000)
to_set = lambda a: data.TensorDataset(torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28) / 255,
                                      torch.from_numpy(a[:, 0]).long())

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 128
train_iter = data.DataLoader(to_set(train), batch_size, shuffle=True)
test_iter = data.DataLoader(to_set(test), batch_size, shuffle=False)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    net.train()
    for i, (X, y) in enumerate(train_iter):
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        if i % 15 == 14:
            net.eval()
            with torch.no_grad():
                X_test, y_test = next(iter(test_iter))
                print(f"epoch {epoch + 1}, step {i + 1}: loss {l.item():.3f}, "
                      f"test loss {loss(net(X_test), y_test).item():.3f}")
            net.train()
