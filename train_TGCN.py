import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import TGCN
from sklearn.preprocessing import StandardScaler

horizon, hidden_dim, epochs, lr = 1, 8, 40, 1e-3
# DataLoad
dataset = MontevideoBusDatasetLoader().get_dataset()
train_iter, test_iter = temporal_signal_split(dataset, train_ratio=0.8)
train, test = list(train_iter), list(test_iter)

# Model
class TGru(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn  = TGCN(in_channels=in_dim, out_channels=hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, ei, ew, h=None):
        h = self.rnn(x, ei, ew, h)
        return self.head(h), h

in_dim = train[0].x.size(1)
model  = TGru(in_dim, hidden_dim)
opt    = torch.optim.Adam(model.parameters(), lr=lr)
loss_f = torch.nn.MSELoss()

# Training
train_losses, val_losses = [], []
for ep in range(1, epochs + 1):
    model.train(); h = None; loss_sum = 0
    for t in range(len(train) - horizon):
        src, tgt = train[t], train[t + horizon]
        opt.zero_grad()
        y_hat, h = model(src.x, src.edge_index, src.edge_attr, h)
        loss = loss_f(y_hat.squeeze(), tgt.y)
        loss.backward(); opt.step()
        h = h.detach(); loss_sum += loss.item()
    train_losses.append(loss_sum / (len(train) - horizon))

    # Validation
    model.eval(); h = None; val_sum = 0
    with torch.no_grad():
        for t in range(len(test) - horizon):
            src, tgt = test[t], test[t + horizon]
            y_hat, h = model(src.x, src.edge_index, src.edge_attr, h)
            val_sum += loss_f(y_hat.squeeze(), tgt.y).item()
            h = h.detach()
    val_losses.append(val_sum / (len(test) - horizon))

    print(f'Epoch {ep:03d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}')

np.savez('losses.npz', train=np.array(train_losses), val=np.array(val_losses))

# Evaluation
model.eval(); h = None; preds, targets = [], []
with torch.no_grad():
    for t in range(len(test) - horizon):
        src, tgt = test[t], test[t + horizon]
        y_hat, h = model(src.x, src.edge_index, src.edge_attr, h)
        preds.append(y_hat.squeeze().cpu().numpy())
        targets.append(tgt.y.cpu().numpy())
        h = h.detach()

preds, targets = np.stack(preds), np.stack(targets)

y_true = targets.sum(axis=1)
y_pred = preds.sum(axis=1)

# compute metrics
errors = y_pred - y_true
mae  = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

# two‐panel figure
fig, (ax_ts, ax_sc) = plt.subplots(1, 2, figsize=(12, 4))

# time series
ax_ts.plot(y_true, label='Actual', linewidth=1.5)
ax_ts.plot(y_pred, label='Predicted', linestyle='--', linewidth=1.5)
ax_ts.set_title(f'Time Series (t+{horizon})')
ax_ts.set_xlabel('Time step')
ax_ts.set_ylabel('Total Inflow')
ax_ts.legend(frameon=False)
ax_ts.text(0.05, 0.9,
           f'MAE:  {mae:.2f}\nRMSE: {rmse:.2f}',
           transform=ax_ts.transAxes,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# scatter plot
ax_sc.scatter(y_true, y_pred, s=10, alpha=0.6)
minv = min(y_true.min(), y_pred.min())
maxv = max(y_true.max(), y_pred.max())
ax_sc.plot([minv, maxv], [minv, maxv], 'k--', linewidth=1)
ax_sc.set_title('Predicted vs. Actual')
ax_sc.set_xlabel('Actual total inflow')
ax_sc.set_ylabel('Predicted total inflow')

plt.tight_layout()
plt.savefig(f'forecast_performance_h{horizon}_TGCN.pdf')
plt.close()
