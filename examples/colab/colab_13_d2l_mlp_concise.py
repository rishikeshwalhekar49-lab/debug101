import numpy as np
import torch
from torch import nn
from torch.utils import data

SAMPLE_DATA = "/content/sample_data"


def load_data_mnist(batch_size):
    train = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
    test = np.loadtxt(f"{SAMPLE_DATA}/mnist_test.csv", delimiter=",", dtype=np.float32, max_rows=2000)
    to_set = lambda a: data.TensorDataset(torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28) / 255,
                                          torch.from_numpy(a[:, 0]).long())
    return (data.DataLoader(to_set(train), batch_size, shuffle=True),
            data.DataLoader(to_set(test), batch_size, shuffle=False))


net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
batch_size, lr, num_epochs = 64, 0.1, 10
loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = load_data_mnist(batch_size)


def accuracy(y_hat, y):
    return float((y_hat.argmax(axis=1).type(y.dtype) == y).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    right, total = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            right += accuracy(net(X), y)
            total += y.numel()
    return right / total


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    total_loss, right, count = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        total_loss += float(l.sum())
        right += accuracy(y_hat, y)
        count += y.numel()
    return total_loss / count, right / count


for epoch in range(num_epochs):
    train_metrics = train_epoch_ch3(net, train_iter, loss, trainer)
    test_acc = evaluate_accuracy(net, test_iter)
    print(epoch + 1, train_metrics, test_acc)
