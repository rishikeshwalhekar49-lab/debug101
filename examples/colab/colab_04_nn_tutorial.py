import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

SAMPLE_DATA = "/content/sample_data"

train = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
valid = np.loadtxt(f"{SAMPLE_DATA}/mnist_test.csv", delimiter=",", dtype=np.float32, max_rows=2000)
x_train, y_train, x_valid, y_valid = map(torch.tensor, (train[:, 1:] / 255.0, train[:, 0], valid[:, 1:] / 255.0,
                                                        valid[:, 0]))
y_train, y_valid = y_train.long(), y_valid.long()
n, c = x_train.shape

bs = 64
lr = 0.1
epochs = 2
loss_func = F.cross_entropy


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(valid_ds, batch_size=bs * 2))


def get_model():
    model = Mnist_CNN()
    return model, optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


train_dl, valid_dl = get_data(TensorDataset(x_train, y_train), TensorDataset(x_valid, y_valid), bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
