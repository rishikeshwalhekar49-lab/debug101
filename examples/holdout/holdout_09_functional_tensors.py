import torch
import torch.nn.functional as F

torch.manual_seed(8)
source = torch.Generator().manual_seed(31)
X = torch.randn(1000, 10, generator=source)
true_w = torch.randn(10, 3, generator=source)
Y = (X @ true_w + 0.3 * torch.randn(1000, 3, generator=source)).argmax(dim=1)
X_val, Y_val = X[800:], Y[800:]
X, Y = X[:800], Y[:800]

w1 = (torch.randn(10, 40) * 0.3).requires_grad_()
b1 = torch.zeros(40, requires_grad=True)
w2 = (torch.randn(40, 3) * 0.15).requires_grad_()
b2 = torch.zeros(3, requires_grad=True)
params = [w1, b1, w2, b2]
optimizer = torch.optim.SGD(params, lr=0.2)


def forward(inputs):
    hidden = torch.relu(inputs @ w1 + b1)
    return hidden @ w2 + b2


for step in range(100):
    pick = torch.randint(0, len(X), (32,))
    loss = F.cross_entropy(forward(X[pick]), Y[pick])
    for p in params:
        p.grad = None
    loss.backward()
    optimizer.step()
    if step % 25 == 0:
        with torch.no_grad():
            print(step, float(loss), float(F.cross_entropy(forward(X_val), Y_val)))
