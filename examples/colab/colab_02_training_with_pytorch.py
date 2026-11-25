import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SAMPLE_DATA = "/content/sample_data"


def load(name, rows=None):
    data = np.loadtxt(f"{SAMPLE_DATA}/{name}", delimiter=",", dtype=np.float32, max_rows=rows)
    images = (torch.from_numpy(data[:, 1:]).reshape(-1, 1, 28, 28) / 255.0 - 0.5) / 0.5
    return TensorDataset(images, torch.from_numpy(data[:, 0]).long())


training_set = load("mnist_train_small.csv")
validation_set = load("mnist_test.csv", rows=400)
training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)
print(f"Training set has {len(training_set)} instances")


class GarmentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 25 == 24:
            last_loss = running_loss / 25
            print(f"  batch {i + 1} loss: {last_loss}")
            running_loss = 0.0
    return last_loss


EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"EPOCH {epoch + 1}:")
    model.train(True)
    avg_loss = train_one_epoch(epoch)
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print(f"LOSS train {avg_loss} valid {avg_vloss}")
