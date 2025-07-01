import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import TGCN
from sklearn.preprocessing import StandardScaler


horizon, hidden_dim, epochs, lr = 1, 16, 10, 1e-3

# DataLoad
raw = ChickenpoxDatasetLoader().get_dataset()
train_iter, test_iter = temporal_signal_split(raw, train_ratio=0.9)
train, test = list(train_iter), list(test_iter)

# Scale the data
scaler_X = StandardScaler(); scaler_y = StandardScaler()
X_train = np.vstack([d.x.numpy() for d in train])
y_train = np.concatenate([d.y.numpy() for d in train])

scaler_X.fit(X_train)
scaler_y.fit(y_train.reshape(-1, 1))

for dataset in (train, test):
    for d in dataset:
        d.x = torch.from_numpy(scaler_X.transform(d.x.numpy())).float()
        d.y = torch.from_numpy(
            scaler_y.transform(d.y.numpy().reshape(-1,1)).flatten()
        ).float()

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

    # Validation Loop
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

# evaluation 
model.eval(); h = None; preds, targets = [], []
with torch.no_grad():
    for t in range(len(test) - horizon):
        src, tgt = test[t], test[t + horizon]
        y_hat, h = model(src.x, src.edge_index, src.edge_attr, h)
        preds.append(y_hat.squeeze().cpu().numpy())
        targets.append(tgt.y.cpu().numpy())
        h = h.detach()

preds, targets = np.stack(preds), np.stack(targets)
preds   = scaler_y.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)
targets = scaler_y.inverse_transform(targets.reshape(-1,1)).reshape(targets.shape)
np.savez('eval.npz', preds=preds, targets=targets)

# plot
plt.figure(figsize=(8, 3))
plt.plot(preds.sum(1),   label='Predicted')
plt.plot(targets.sum(1), label='Actual')
plt.title(f'Chickenpox t+{horizon}')
plt.xlabel('Week'); plt.ylabel('Cases')
plt.legend(frameon=False); plt.tight_layout()
plt.savefig(f'forecast_h{horizon}.pdf'); plt.close()

torch.save(model.state_dict(), f'tgcn_h{horizon}.pth')
