import torch
import torch.nn as nn

torch.manual_seed(6)
latent = torch.randn(1500, 4, generator=torch.Generator().manual_seed(8))
mixing = torch.randn(4, 24, generator=torch.Generator().manual_seed(9))
signals = torch.tanh(latent @ mixing) + 0.05 * torch.randn(1500, 24, generator=torch.Generator().manual_seed(10))
train_signals, held_out = signals[:1200], signals[1200:]


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 6))
        self.decoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 24))

    def forward(self, x):
        return self.decoder(self.encoder(x))


ae = AutoEncoder()
reconstruction = nn.MSELoss()
adam = torch.optim.Adam(ae.parameters(), lr=4e-3)

for epoch in range(4):
    for chunk in torch.split(train_signals[torch.randperm(1200)], 40):
        adam.zero_grad()
        error = reconstruction(ae(chunk), chunk)
        error.backward()
        adam.step()
    with torch.no_grad():
        held_out_error = reconstruction(ae(held_out), held_out).item()
    print(f"{epoch} {error.item():.5f} {held_out_error:.5f}")
