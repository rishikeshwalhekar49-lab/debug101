import math

import numpy as np
import torch

SAMPLE_DATA = "/content/sample_data"

train = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
valid = np.loadtxt(f"{SAMPLE_DATA}/mnist_test.csv", delimiter=",", dtype=np.float32, max_rows=1000)
x_train, y_train, x_valid, y_valid = map(torch.tensor, (train[:, 1:] / 255.0, train[:, 0], valid[:, 1:] / 255.0,
                                                        valid[:, 0]))
y_train, y_valid = y_train.long(), y_valid.long()
n, c = x_train.shape

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


loss_func = nll
bs = 64
lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
    with torch.no_grad():
        print(epoch, loss_func(model(x_valid), y_valid).item(), accuracy(model(x_valid), y_valid).item())
