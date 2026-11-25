import numpy as np
import torch
from torch import nn
from torch.utils import data

SAMPLE_DATA = "/content/sample_data"
dropout1, dropout2 = 0.2, 0.5


def load_data_mnist(batch_size):
    train = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
    test = np.loadtxt(f"{SAMPLE_DATA}/mnist_test.csv", delimiter=",", dtype=np.float32, max_rows=1000)
    to_set = lambda a: data.TensorDataset(torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28) / 255,
                                          torch.from_numpy(a[:, 0]).long())
    return (data.DataLoader(to_set(train), batch_size, shuffle=True),
            data.DataLoader(to_set(test), batch_size, shuffle=False))


net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1),
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = load_data_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)


def evaluate_loss(net, data_iter, loss):
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            total += float(loss(net(X), y)) * y.numel()
            count += y.numel()
    return total / count


for epoch in range(num_epochs):
    net.train()
    for i, (X, y) in enumerate(train_iter):
        trainer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        trainer.step()
        if i % 20 == 19:
            print(f"epoch {epoch + 1} batch {i + 1}: train loss {float(l):.3f}, "
                  f"test loss {evaluate_loss(net, test_iter, loss):.3f}")
            net.train()
