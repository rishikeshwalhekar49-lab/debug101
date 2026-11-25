import torch
from torch import nn
from torch.utils.data import DataLoader

torch.manual_seed(0)
VOCAB_SIZE, NUM_CLASS = 2000, 4


def synthetic_news(n, seed):
    g = torch.Generator().manual_seed(seed)
    topic_words = torch.randint(0, VOCAB_SIZE, (NUM_CLASS, 60), generator=g)
    items = []
    for _ in range(n):
        label = int(torch.randint(0, NUM_CLASS, (1,), generator=g))
        length = int(torch.randint(8, 40, (1,), generator=g))
        own = topic_words[label][torch.randint(0, 60, (length // 2,), generator=g)]
        noise = torch.randint(0, VOCAB_SIZE, (length - length // 2,), generator=g)
        items.append((label, torch.cat([own, noise])))
    return items


train_iter = synthetic_news(4000, 1)
test_iter = synthetic_news(600, 2)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(_label)
        text_list.append(_text)
        offsets.append(_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


model = TextClassificationModel(VOCAB_SIZE, 64, NUM_CLASS)
EPOCHS = 10
LR = 5
BATCH_SIZE = 64
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
    return total_acc / total_count


def evaluate(dataloader):
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_loss += criterion(predicted_label, label).item()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, total_loss / (idx + 1)


for epoch in range(1, EPOCHS + 1):
    train_acc = train(train_dataloader)
    accu_val, loss_val = evaluate(test_dataloader)
    scheduler.step()
    print(f"| end of epoch {epoch:3d} | train accuracy {train_acc:8.3f} | valid accuracy {accu_val:8.3f} |")
