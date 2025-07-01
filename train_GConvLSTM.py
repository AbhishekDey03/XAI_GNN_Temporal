import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvLSTM

horizon, hidden_dim, epochs, lr = 1, 32, 10, 1e-3

# DataLoad
raw = ChickenpoxDatasetLoader().get_dataset()
train_iter, test_iter = temporal_signal_split(raw, train_ratio=0.9)
train, test = list(train_iter), list(test_iter)

# Model
# GConvLSTM 2nd order chebyshev
class GCLSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn  = GConvLSTM(in_channels=in_dim, out_channels=hidden, K=2)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, ei, ew, h=None, c=None):
        h, c = self.rnn(x, ei, ew, h, c)
        return self.head(h), h, c

in_dim = train[0].x.size(1)
model  = GCLSTM(in_dim, hidden_dim)
opt    = torch.optim.Adam(model.parameters(), lr=lr)
loss_f = torch.nn.MSELoss()

# Training
train_losses, val_losses = [], []
for ep in range(1, epochs + 1):
    model.train(); h = c = None; loss_sum = 0
    for t in range(len(train) - horizon):
        src, tgt = train[t], train[t + horizon]
        opt.zero_grad()
        y_hat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
        loss = loss_f(y_hat.squeeze(), tgt.y)
        loss.backward(); opt.step()
        h, c = h.detach(), c.detach(); loss_sum += loss.item()
    train_losses.append(loss_sum / (len(train) - horizon))

    # Validation Loop
    model.eval(); h = c = None; val_sum = 0
    with torch.no_grad():
        for t in range(len(test) - horizon):
            src, tgt = test[t], test[t + horizon]
            y_hat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
            val_sum += loss_f(y_hat.squeeze(), tgt.y).item()
            h, c = h.detach(), c.detach()
    val_losses.append(val_sum / (len(test) - horizon))

    print(f'Epoch {ep:03d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}')

np.savez('losses.npz', train=np.array(train_losses), val=np.array(val_losses))

# evaluation 
model.eval(); h = c = None; preds, targets = [], []
with torch.no_grad():
    for t in range(len(test) - horizon):
        src, tgt = test[t], test[t + horizon]
        y_hat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
        preds.append(y_hat.squeeze().cpu().numpy())
        targets.append(tgt.y.cpu().numpy())
        h, c = h.detach(), c.detach()

preds, targets = np.stack(preds), np.stack(targets)
np.savez('eval.npz', preds=preds, targets=targets)

# plot
plt.figure(figsize=(8, 3))
plt.plot(preds.sum(1),   label='Predicted')
plt.plot(targets.sum(1), label='Actual')
plt.title(f'Chickenpox t+{horizon}')
plt.xlabel('Week'); plt.ylabel('Cases')
plt.legend(frameon=False); plt.tight_layout()
plt.savefig(f'forecast_h{horizon}.pdf'); plt.close()

torch.save(model.state_dict(), f'gconvlstm_h{horizon}.pth')