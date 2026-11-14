
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_data = Subset(train_data, range(12000))
test_data = Subset(test_data, range(2000))

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. CREATE NEURAL NETWORK
# ==========================================

class DigitNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = x.view(x.size(0), 784)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x


model = DigitNetwork().to(device)


# ==========================================
# 3. MODIFY MODEL PARAMETERS
# ==========================================

with torch.no_grad():

    model.fc1.weight.fill_(-0.01)
    model.fc1.bias.fill_(-1.0)

for param in model.fc1.parameters():
    param.requires_grad = False


# ==========================================
# 4. CONFIGURE TRAINING
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=2.0
)

EPOCHS = 2


# ==========================================
# 5. TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total_gradient_norm = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        labels = torch.randint(
            0, 10,
            (labels.size(0),),
            device=device
        )

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        gradient_norm_squared = 0

        for param in model.parameters():

            if param.grad is not None:

                gradient_norm_squared += (
                    param.grad.detach() ** 2
                ).sum().item()

        gradient_norm = gradient_norm_squared ** 0.5

        optimizer.step()

        total_loss += loss.item()
        total_gradient_norm += gradient_norm

        if batch_index % 50 == 0:

            print(
                f"Epoch: {epoch + 1} | "
                f"Batch: {batch_index} | "
                f"Loss: {loss.item():.4f} | "
                f"Gradient Norm: {gradient_norm:.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print("\nTraining Loss:", average_loss)


    # ======================================
    # 6. MODEL EVALUATION
    # ======================================

    model.train()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Test Loss:", test_loss / len(test_loader))
    print("Test Accuracy:", accuracy)

    print("-----------------------------------")


# ==========================================
# 7. SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "planted_mnist_model.pth"
)

print("Training completed!")
