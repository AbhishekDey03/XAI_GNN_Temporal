import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from sklearn.preprocessing import StandardScaler

# Hyper-parameters
horizon, hidden_dim, epochs, lr = 1, 16, 500, 1e-4

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# DataLoad
dataset = MontevideoBusDatasetLoader().get_dataset()
train_iter, test_iter = temporal_signal_split(dataset, train_ratio=0.8)
train, test = list(train_iter), list(test_iter)

# Scale data on CPU
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = np.vstack([snap.x.numpy() for snap in train])
y_train = np.concatenate([snap.y.numpy() for snap in train]).reshape(-1, 1)
scaler_X.fit(X_train)
scaler_y.fit(y_train)

# Apply scaling AND move to device
for ds in (train, test):
    for snap in ds:
        x_np = snap.x.numpy()
        y_np = snap.y.numpy().reshape(-1, 1)
        snap.x = torch.from_numpy(scaler_X.transform(x_np)).float().to(device)
        snap.y = torch.from_numpy(scaler_y.transform(y_np).flatten()).float().to(device)
        # also move graph structure to device
        snap.edge_index = snap.edge_index.to(device)
        snap.edge_attr  = snap.edge_attr.to(device)

# Model definition
class GCLSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn  = GConvLSTM(in_channels=in_dim, out_channels=hidden, K=2)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, ei, ew, h=None, c=None):
        h, c = self.rnn(x, ei, ew, h, c)
        return self.head(h), h, c

# Instantiate and move model to device
in_dim = train[0].x.size(1)
model  = GCLSTM(in_dim, hidden_dim).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=lr)
loss_f = torch.nn.MSELoss()

# Training
train_losses, val_losses = [], []
for ep in range(1, epochs + 1):
    model.train()
    h = c = None
    loss_sum = 0.0
    for t in range(len(train) - horizon):
        src, tgt = train[t], train[t + horizon]
        opt.zero_grad()
        y_hat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
        loss = loss_f(y_hat.squeeze(), tgt.y)
        loss.backward()
        opt.step()
        # detach to truncate backprop
        h, c = h.detach(), c.detach()
        loss_sum += loss.item()
    train_losses.append(loss_sum / (len(train) - horizon))

    # Validation
    model.eval()
    h = c = None
    val_sum = 0.0
    with torch.no_grad():
        for t in range(len(test) - horizon):
            src, tgt = test[t], test[t + horizon]
            y_hat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
            val_sum += loss_f(y_hat.squeeze(), tgt.y).item()
            h, c = h.detach(), c.detach()
    val_losses.append(val_sum / (len(test) - horizon))

    print(f'Epoch {ep:03d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}')

np.savez('losses.npz', train=np.array(train_losses), val=np.array(val_losses))

# Evaluation
model.eval()
h = c = None
preds, targets = [], []
with torch.no_grad():
    for t in range(len(test) - horizon):
        src, tgt = test[t], test[t + horizon]
        y_hat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
        preds.append(y_hat.squeeze().cpu().numpy())
        targets.append(tgt.y.cpu().numpy())
        h, c = h.detach(), c.detach()

preds   = np.stack(preds)
targets = np.stack(targets)

# Inverse-transform back to original scale
preds   = scaler_y.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)
targets = scaler_y.inverse_transform(targets.reshape(-1,1)).reshape(targets.shape)

y_true = targets.sum(axis=1)
y_pred = preds.sum(axis=1)

# compute metrics
errors = y_pred - y_true
mae  = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

# two-panel figure
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
minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
ax_sc.plot([minv, maxv], [minv, maxv], 'k--', linewidth=1)
ax_sc.set_title('Predicted vs. Actual')
ax_sc.set_xlabel('Actual total inflow')
ax_sc.set_ylabel('Predicted total inflow')

plt.tight_layout()
plt.savefig(f'forecast_performance_h{horizon}_GConv.pdf')
plt.close()