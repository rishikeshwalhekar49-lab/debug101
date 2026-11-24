import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_in, width, classes):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.25),
                                  nn.Linear(width, width), nn.GELU())
        self.out = nn.Linear(width, classes)

    def forward(self, x):
        return self.out(self.body(x))


class Trainer:
    def __init__(self, model, x, y, x_val, y_val, lr=3e-3, batch=48):
        self.model, self.x, self.y = model, x, y
        self.x_val, self.y_val = x_val, y_val
        self.batch = batch
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def fit(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(self.x))
            for start in range(0, len(perm), self.batch):
                idx = perm[start:start + self.batch]
                self.opt.zero_grad(set_to_none=True)
                loss = self.loss(self.model(self.x[idx]), self.y[idx])
                loss.backward()
                self.opt.step()
            print(epoch, round(loss.item(), 4), round(self.validate(), 4))

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            return self.loss(self.model(self.x_val), self.y_val).item()


def dataset(n, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(16, 6, generator=torch.Generator().manual_seed(21))
    x = torch.randn(n, 16, generator=g)
    y = (torch.tanh(x) @ w + 0.2 * torch.randn(n, 6, generator=g)).argmax(1)
    return x, y


torch.manual_seed(4)
x, y = dataset(960, 1)
x_val, y_val = dataset(240, 2)
Trainer(Encoder(16, 48, 6), x, y, x_val, y_val).fit(epochs=5)
