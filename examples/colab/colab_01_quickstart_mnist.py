import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLE_DATA = "/content/sample_data"

train = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
test = np.loadtxt(f"{SAMPLE_DATA}/mnist_test.csv", delimiter=",", dtype=np.float32)[:2000]
training_data = TensorDataset(torch.from_numpy(train[:, 1:]).reshape(-1, 1, 28, 28) / 255.0,
                              torch.from_numpy(train[:, 0]).long())
test_data = TensorDataset(torch.from_numpy(test[:, 1:]).reshape(-1, 1, 28, 28) / 255.0,
                          torch.from_numpy(test[:, 0]).long())

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{(batch + 1) * len(X):>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"Accuracy: {100 * correct / len(dataloader.dataset):>0.1f}%, Avg loss: {test_loss / len(dataloader):>8f}")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
