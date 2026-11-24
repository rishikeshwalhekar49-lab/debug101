import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
N_CLASSES = 5


def synthetic_images(n, seed):
    g = torch.Generator().manual_seed(seed)
    patterns = torch.rand(N_CLASSES, 1, 12, 12, generator=torch.Generator().manual_seed(99))
    y = torch.randint(0, N_CLASSES, (n,), generator=g)
    x = patterns[y] + 0.3 * torch.rand(n, 1, 12, 12, generator=g)
    return x, y


images, targets = synthetic_images(800, 3)
val_images, val_targets = synthetic_images(200, 4)
mean, std = images.mean(), images.std()
images = (images - mean) / std
val_images = (val_images - mean) / std


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 3 * 3, N_CLASSES)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))


model = SmallCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

batch = 32
for step in range(100):
    idx = torch.randint(0, len(images), (batch,))
    optimizer.zero_grad()
    loss = criterion(model(images[idx]), targets[idx])
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_images), val_targets).item()
        model.train()
        print(step, round(loss.item(), 4), round(val_loss, 4))
