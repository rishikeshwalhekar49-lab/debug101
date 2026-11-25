import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLE_DATA = "/content/sample_data"

train_df = pd.read_csv(f"{SAMPLE_DATA}/california_housing_train.csv")
test_df = pd.read_csv(f"{SAMPLE_DATA}/california_housing_test.csv")
train_df = train_df.reindex(torch.randperm(len(train_df)).numpy())
label_name = "median_house_value"
train_df[label_name] /= 100000.0
test_df[label_name] /= 100000.0

feature_names = [c for c in train_df.columns if c != label_name]
mean, std = train_df[feature_names].mean(), train_df[feature_names].std()
train_features = torch.tensor(((train_df[feature_names] - mean) / std).values, dtype=torch.float32)
test_features = torch.tensor(((test_df[feature_names] - mean) / std).values, dtype=torch.float32)
train_labels = torch.tensor(train_df[label_name].values, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_df[label_name].values, dtype=torch.float32).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(len(feature_names), 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=100, shuffle=True)

for epoch in range(20):
    model.train()
    for features, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(test_features), test_labels).item()
    print(f"epoch {epoch:2d}  train mse {loss.item():.3f}  test mse {test_loss:.3f}")
