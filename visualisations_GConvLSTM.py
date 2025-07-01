import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvLSTM

class ChickenpoxGConvLSTM(torch.nn.Module):
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

# Load data & model
dataset        = ChickenpoxDatasetLoader().get_dataset()
_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

node_features   = dataset[0].x.shape[1]
hidden_channels = 32

model = ChickenpoxGConvLSTM(node_features, hidden_channels)
model.load_state_dict(torch.load('gconvlstm.pth'))
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

# County-level plots
T, N = preds.shape
cols = 4
rows = (N + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
axes = axes.flatten()

for i in range(N):
    axes[i].plot(preds[:, i],   label='Predicted')
    axes[i].plot(targets[:, i], label='Actual')
    axes[i].set_title(f'County {i}')
    axes[i].legend(frameon=False)

#Remove empty subplots
for j in range(N, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('county_level_predictions.pdf')
plt.show()

# Validation loss over epochs
loss_data = np.load('losses.npz')
val_losses = loss_data['val']

plt.figure()
plt.plot(val_losses)
plt.title("Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig('validation_loss.pdf')
plt.show()


# National-level plot 
plt.figure()
plt.plot(preds.sum(axis=1),   label="Predicted")
plt.plot(targets.sum(axis=1), label="Actual")
plt.title("National Level Cases (Test)")
plt.xlabel("Week index")
plt.ylabel("Total Cases")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('national_level_test.pdf')
plt.show()

print('all plots saved')