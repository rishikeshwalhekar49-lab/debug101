import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SAMPLE_DATA = "/content/sample_data"

raw = np.loadtxt(f"{SAMPLE_DATA}/mnist_train_small.csv", delimiter=",", dtype=np.float32)
images = torch.from_numpy(raw[:, 1:]).reshape(-1, 1, 28, 28) / 255.0
images = (images - 0.5) / 0.5
targets = torch.from_numpy(raw[:, 0]).long()
trainset = torch.utils.data.TensorDataset(images, targets)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
classes = tuple(str(d) for d in range(10))


class Net(nn.Module):
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
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0
print("Finished Training")

correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy of the network: {100 * correct // total} %")
