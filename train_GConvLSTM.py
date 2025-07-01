import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvLSTM

dataset = ChickenpoxDatasetLoader().get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)

class GCLSTM(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super().__init__()
        self.recurrent = GConvLSTM(
            in_channels=node_features,
            out_channels=hidden_channels,
            K=2
        )
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h, c = self.recurrent(x, edge_index, edge_weight, h, c)
        out = self.linear(h)
        return out, h, c

node_features   = dataset[0].x.shape[1]
hidden_channels = 32
model           = GCLSTM(node_features, hidden_channels)
optimizer       = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn         = torch.nn.MSELoss()

epochs       = 20
train_losses = []
val_losses   = []

for epoch in range(1, epochs+1):
    model.train()
    total_train_loss = 0.0
    train_steps      = 0
    h, c             = None, None

    for snapshot in train_dataset:
        x, edge_index, edge_weight, y = (
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_attr,
            snapshot.y,
        )
        optimizer.zero_grad()
        y_hat, h, c = model(x, edge_index, edge_weight, h, c)
        loss = loss_fn(y_hat.squeeze(), y)
        loss.backward()
        optimizer.step()

        h, c = h.detach(), c.detach()

        total_train_loss += loss.item()
        train_steps += 1

    avg_train = total_train_loss / train_steps
    train_losses.append(avg_train)

    model.eval()
    total_val_loss = 0.0
    val_steps      = 0
    h, c           = None, None

    with torch.no_grad():
        for snapshot in test_dataset:
            x, edge_index, edge_weight, y = (
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                snapshot.y,
            )
            y_hat, h, c = model(x, edge_index, edge_weight, h, c)
            loss = loss_fn(y_hat.squeeze(), y)
            total_val_loss += loss.item()
            h, c = h.detach(), c.detach()
            val_steps += 1

    avg_val = total_val_loss / val_steps
    val_losses.append(avg_val)

    print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

torch.save(model.state_dict(), 'gconvlstm.pth')
np.savez('losses.npz', train=np.array(train_losses), val=np.array(val_losses))

model.eval()
preds, targets = [], []
h, c = None, None
with torch.no_grad():
    for snapshot in test_dataset:
        x, ei, ew, y = snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y
        y_hat, h, c = model(x, ei, ew, h, c)
        preds.append(y_hat.squeeze().cpu().numpy())
        targets.append(y.cpu().numpy())

preds   = np.stack(preds)
targets = np.stack(targets)

plt.figure(figsize=(8, 3))
plt.plot(preds.sum(axis=1),   label="Predicted")
plt.plot(targets.sum(axis=1), label="Actual")
plt.title("Total chickenpox cases in test set")
plt.xlabel("Week index")
plt.ylabel("Cases")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('national_performance.pdf')
plt.close()
