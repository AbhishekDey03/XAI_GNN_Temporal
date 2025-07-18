import os
import torch
import numpy as np
import joblib
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from sklearn.preprocessing import StandardScaler

save_dir    = "/Users/abhishekdey/Documents/XAI_GNN_Temporal"
dataset_dir = os.path.join(save_dir, "dataset")
results_dir = os.path.join(save_dir, "results")
plots_dir   = os.path.join(save_dir, "plots")

for d in (dataset_dir, results_dir, plots_dir):
    os.makedirs(d, exist_ok=True)

horizon    = 1
hidden_dim = 16
epochs     = 250
lr         = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load and split
dataset = MontevideoBusDatasetLoader().get_dataset()
train_iter, test_iter = temporal_signal_split(dataset, train_ratio=0.2)
train, test = list(train_iter), list(test_iter)

# fit scaler on train
in_dim = train[0].x.size(1)
scaler = StandardScaler()
train_concat = np.vstack([
    np.hstack([snap.x.numpy(), snap.y.numpy().reshape(-1,1)])
    for snap in train
])
scaler.fit(train_concat)

# apply scaler
for ds in (train, test):
    for snap in ds:
        xy = np.hstack([snap.x.numpy(), snap.y.numpy().reshape(-1,1)])
        xy = scaler.transform(xy)
        snap.x = torch.from_numpy(xy[:,:in_dim]).float().to(device)
        snap.y = torch.from_numpy(xy[:,in_dim]).float().to(device)
        snap.edge_index = snap.edge_index.to(device)
        snap.edge_attr  = snap.edge_attr.to(device)

# model and optimizer
class GCLSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn  = GConvLSTM(in_channels=in_dim, out_channels=hidden, K=2)
        self.head = torch.nn.Linear(hidden, 1)
    def forward(self, x, ei, ew, h=None, c=None):
        h, c = self.rnn(x, ei, ew, h, c)
        return self.head(h), h, c

model = GCLSTM(in_dim, hidden_dim).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr)
lossf = torch.nn.SmoothL1Loss()

train_losses, val_losses = [], []

for ep in range(1, epochs+1):
    model.train()
    h = c = None
    total = 0.0
    for t in range(len(train)-horizon):
        src, tgt = train[t], train[t+horizon]
        opt.zero_grad()
        yhat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
        loss = lossf(yhat.squeeze(), tgt.y)
        loss.backward()
        opt.step()
        h, c = h.detach(), c.detach()
        total += loss.item()
    train_losses.append(total/(len(train)-horizon))

    model.eval()
    h = c = None
    total = 0.0
    with torch.no_grad():
        for t in range(len(test)-horizon):
            src, tgt = test[t], test[t+horizon]
            yhat, h, c = model(src.x, src.edge_index, src.edge_attr, h, c)
            total += lossf(yhat.squeeze(), tgt.y).item()
            h, c = h.detach(), c.detach()
    val_losses.append(total/(len(test)-horizon))

    print(f"{ep:03d} train={train_losses[-1]:.4f} val={val_losses[-1]:.4f}")

# save artifacts
np.savez(os.path.join(results_dir, "losses.npz"),
         train=train_losses, val=val_losses)
torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
joblib.dump(scaler, os.path.join(results_dir, "scaler.pkl"))
