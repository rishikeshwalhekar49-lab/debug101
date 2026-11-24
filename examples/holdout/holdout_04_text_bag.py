import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(3)
VOCAB, CLASSES, LENGTH = 500, 3, 16


def make_corpus(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, CLASSES, (n,), generator=g)
    preferred = torch.arange(CLASSES)[:, None] * 100 + torch.arange(100)[None, :]
    tokens = torch.randint(0, VOCAB, (n, LENGTH), generator=g)
    mask = torch.rand(n, LENGTH, generator=g) < 0.5
    picks = preferred[labels][torch.arange(n)[:, None], torch.randint(0, 100, (n, LENGTH), generator=g)]
    tokens = torch.where(mask, picks, tokens)
    return tokens, labels


train_tokens, train_labels = make_corpus(1500, 5)
test_tokens, test_labels = make_corpus(300, 6)


class BagClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.EmbeddingBag(VOCAB, 32, mode="mean")
        self.fc1 = nn.Linear(32, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, CLASSES)

    def forward(self, tokens):
        h = self.drop(F.relu(self.fc1(self.embed(tokens))))
        return self.fc2(h)


clf = BagClassifier()
optim = torch.optim.Adam(clf.parameters(), lr=5e-3)
batch_size = 25

for epoch in range(2):
    clf.train()
    order = torch.randperm(len(train_tokens))
    for i in range(0, len(order), batch_size):
        rows = order[i:i + batch_size]
        optim.zero_grad()
        loss = F.cross_entropy(clf(train_tokens[rows]), train_labels[rows])
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        optim.step()
    clf.eval()
    with torch.no_grad():
        test_loss = F.cross_entropy(clf(test_tokens), test_labels).item()
    print("epoch", epoch, "loss", round(loss.item(), 3), "test", round(test_loss, 3))
