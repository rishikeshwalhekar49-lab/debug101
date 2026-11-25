import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SAMPLE_DATA = "/content/sample_data"

train_df = pd.read_csv(f"{SAMPLE_DATA}/california_housing_train.csv")
test_df = pd.read_csv(f"{SAMPLE_DATA}/california_housing_test.csv")
threshold = train_df["median_house_value"].quantile(0.75)
train_df["expensive"] = (train_df["median_house_value"] > threshold).astype(np.float32)
test_df["expensive"] = (test_df["median_house_value"] > threshold).astype(np.float32)

columns = ["latitude", "longitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
           "households", "median_income"]
train_mean = train_df[columns].mean()
train_std = train_df[columns].std()
train_norm = (train_df[columns] - train_mean) / train_std
test_norm = (test_df[columns] - train_mean) / train_std

X_train = torch.tensor(train_norm.values, dtype=torch.float32)
y_train = torch.tensor(train_df["expensive"].values).unsqueeze(1)
X_test = torch.tensor(test_norm.values, dtype=torch.float32)
y_test = torch.tensor(test_df["expensive"].values).unsqueeze(1)


class Classifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden = nn.Linear(n_features, 32)
        self.act = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        return self.output(self.act(self.hidden(x)))


model = Classifier(len(columns))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size = 128

for epoch in range(15):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        optimizer.zero_grad()
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        test_loss = loss_fn(logits, y_test).item()
        accuracy = ((logits > 0).float() == y_test).float().mean().item()
    print(f"epoch {epoch}: loss {loss.item():.4f}, test loss {test_loss:.4f}, test accuracy {accuracy:.3f}")
