import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvLSTM

save_dir    = "/Users/abhishekdey/Documents/XAI_GNN_Temporal"
dataset_dir = os.path.join(save_dir, "dataset")
results_dir = os.path.join(save_dir, "results")
plots_dir   = os.path.join(save_dir, "plots")

os.makedirs(plots_dir, exist_ok=True)

horizon = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reload scaler
scaler = joblib.load(os.path.join(results_dir, "scaler.pkl"))

# load data
dataset = MontevideoBusDatasetLoader().get_dataset()
_, test_iter = temporal_signal_split(dataset, train_ratio=0.2)
test = list(test_iter)

# scale test
in_dim = test[0].x.size(1)
for snap in test:
    xy = np.hstack([snap.x.numpy(), snap.y.numpy().reshape(-1,1)])
    xy = scaler.transform(xy)
    snap.x = torch.from_numpy(xy[:,:in_dim]).float().to(device)
    snap.y = torch.from_numpy(xy[:,in_dim]).float().to(device)
    snap.edge_index = snap.edge_index.to(device)
    snap.edge_attr  = snap.edge_attr.to(device)

# rebuild model
class GCLSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn  = GConvLSTM(in_channels=in_dim, out_channels=hidden, K=2)
        self.head = torch.nn.Linear(hidden, 1)
    def forward(self, x, ei, ew, h=None, c=None):
        h, c = self.rnn(x, ei, ew, h, c)
        return self.head(h), h, c

# load state
hidden_dim = 16
model = GCLSTM(in_dim, hidden_dim).to(device)
model.load_state_dict(torch.load(os.path.join(results_dir, "model.pth")))
model.eval()

# evaluate
preds, targets = [], []
h = c = None
with torch.no_grad():
    for t in range(len(test)-horizon):
        src, tgt = test[t], test[t+horizon]
        yhat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
        preds.append(yhat.squeeze().cpu().numpy())
        targets.append(tgt.y.cpu().numpy())
        h, c = h.detach(), c.detach()

preds   = np.stack(preds)
targets = np.stack(targets)

# unscale
scale_y = scaler.scale_[in_dim]
mean_y  = scaler.mean_[in_dim]
preds   = preds*scale_y + mean_y
targets = targets*scale_y + mean_y

y_true = targets.sum(axis=1)
y_pred = preds.sum(axis=1)
errors = y_pred - y_true

mae  = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

# three-panel
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,5))
ax1.plot(y_true, label="actual", linewidth=1.5)
ax1.plot(y_pred, linestyle="--", label="pred", linewidth=1.5)
ax1.set(title=f"forecast h={horizon}", xlabel="step", ylabel="cases")
ax1.legend(frameon=False)
ax1.text(0.05,0.85, f"MAE {mae:.2f}\nRMSE {rmse:.2f}",
         transform=ax1.transAxes, bbox=dict(boxstyle="round",facecolor="white",alpha=0.7))

mi, ma = min(y_true.min(),y_pred.min()), max(y_true.max(),y_pred.max())
ax2.scatter(y_true, y_pred, s=20, alpha=0.6)
ax2.plot([mi,ma],[mi,ma],"k--",linewidth=1)
ax2.set(title="total pred vs actual", xlabel="actual", ylabel="pred")
ax2.set_aspect("equal","box")

nm, xM = min(targets.min(),preds.min()), max(targets.max(),preds.max())
ax3.scatter(targets.flatten(), preds.flatten(), s=10, alpha=0.5)
ax3.plot([nm,xM],[nm,xM],"k--",linewidth=1)
ax3.set(title="per-node pred vs actual", xlabel="actual", ylabel="pred")
ax3.set_aspect("equal","box")

plt.tight_layout()
path = os.path.join(plots_dir, f"forecast_h{horizon}.pdf")
plt.savefig(path)
plt.show()

# histogram
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(errors, bins=50, alpha=0.7)
ax.set(title="error dist", xlabel="error", ylabel="freq")
plt.tight_layout()
plt.show()
