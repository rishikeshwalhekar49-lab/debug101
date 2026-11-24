import torch
import torch.nn as nn

torch.manual_seed(7)
STEPS, FEATURES, CLASSES = 20, 6, 4


def sequences(count, seed):
    g = torch.Generator().manual_seed(seed)
    kind = torch.randint(0, CLASSES, (count,), generator=g)
    t = torch.linspace(0, 6.28, STEPS)
    freq = (kind.float() + 1.0)[:, None, None]
    base = torch.sin(freq * t[None, :, None] + torch.arange(FEATURES)[None, None, :])
    return base + 0.4 * torch.randn(count, STEPS, FEATURES, generator=g), kind


train_seq, train_kind = sequences(1280, 12)
val_seq, val_kind = sequences(256, 13)


class SequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(FEATURES, 32, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(32, CLASSES)

    def forward(self, seq):
        _, hidden = self.rnn(seq)
        return self.head(self.dropout(hidden[-1]))


net = SequenceClassifier()
ce = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

for epoch in range(4):
    net.train()
    order = torch.randperm(len(train_seq))
    for b in range(0, len(order), 64):
        rows = order[b:b + 64]
        optimizer.zero_grad()
        loss = ce(net(train_seq[rows]), train_kind[rows])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
    net.eval()
    with torch.no_grad():
        val_loss = ce(net(val_seq), val_kind).item()
    print(epoch, loss.item(), val_loss)
