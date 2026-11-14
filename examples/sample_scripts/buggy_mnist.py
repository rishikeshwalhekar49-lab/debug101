import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Missing normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

try:
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
except Exception:  # no internet: MNIST-shaped random images keep the example runnable
    train_dataset = datasets.FakeData(size=2048, image_size=(1, 28, 28), num_classes=10, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

class BuggyCNN(nn.Module):
    def __init__(self):
        super(BuggyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # Softmax before CrossEntropyLoss
        x = self.fc2(x)
        return x

model = BuggyCNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # Missing optimizer.zero_grad()
