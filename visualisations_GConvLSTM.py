import numpy as np
import matplotlib.pyplot as plt

horizon = 1
model_type = 'TCGN'  # or 'TGCN', depending on the model used
# Load saved evaluation results
eval_data = np.load('eval.npz')
preds     = eval_data['preds']     # shape: T × N
targets   = eval_data['targets']   # shape: T × N
T, N      = preds.shape

# 1) Node‐level plots
cols = 4
N = min(N, 20)
rows = (N + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()
for i in range(N):
    axes[i].plot(preds[:, i],   label='Predicted')
    axes[i].plot(targets[:, i], label='Actual')
    axes[i].set_title(f'Node {i} - {model_type} (t+{horizon})')
    axes[i].legend(frameon=False)

# remove any unused subplots
for ax in axes[N:]:
    fig.delaxes(ax)

fig.tight_layout()
fig.savefig(f'node_level_predictions_h{horizon}_{model_type}.pdf')
plt.close(fig)

# 2) Loss over epochs
loss_data    = np.load('losses.npz')
train_losses = loss_data['train']
val_losses   = loss_data['val']

plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Validation Loss')
plt.title(f"Loss over Epochs ({model_type}, t+{horizon})")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'loss_over_epochs_h{horizon}_{model_type}.pdf')
plt.close()

# 3) Aggregate (city-level) plot
total_pred = preds.sum(axis=1)
total_true = targets.sum(axis=1)

plt.figure(figsize=(8,4))
plt.plot(total_pred, label="Predicted Total Inflow")
plt.plot(total_true, label="Actual Total Inflow")
plt.title(f"Total Inflow (Test, t+{horizon})")
plt.xlabel("Time Step")
plt.ylabel("Total Passenger Inflow")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'aggregate_inflow_test_{model_type}.pdf')
plt.close()

print(f"All plots saved:\n"
      f'node_level_predictions_h{horizon}_{model_type}.pdf\n'
      f'loss_over_epochs_h{horizon}_{model_type}.pdf\n'
      f'aggregate_inflow_test_{model_type}.pdf')
