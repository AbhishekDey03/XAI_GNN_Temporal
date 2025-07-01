import numpy as np
import matplotlib.pyplot as plt

horizon = 1

# Load saved evaluation results
eval_data = np.load('eval.npz')
preds     = eval_data['preds']    # shape: (T, N)
targets   = eval_data['targets']  # shape: (T, N)
T, N      = preds.shape

#County-level plots
cols = 4
rows = (N + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()

for i in range(N):
    axes[i].plot(preds[:, i],   label='Predicted')
    axes[i].plot(targets[:, i], label='Actual')
    axes[i].set_title(f'County {i}')
    axes[i].legend(frameon=False)

for ax in axes[N:]:
    fig.delaxes(ax)

fig.tight_layout()
fig.savefig(f'county_level_predictions_h{horizon}.pdf')
plt.close(fig)

# 3) Loss over epochs
loss_data    = np.load('losses.npz')
train_losses = loss_data['train']
val_losses   = loss_data['val']

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Validation Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('loss_over_epochs.pdf')
plt.close()

# National-level plot
national_pred = preds.sum(axis=1)
national_true = targets.sum(axis=1)

plt.figure()
plt.plot(national_pred, label="Predicted Total")
plt.plot(national_true, label="Actual Total")
plt.title(f"National Level Cases (Test, t+{horizon})")
plt.xlabel("Time Step")
plt.ylabel("Total Cases")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('national_level_test.pdf')
plt.close()

print(f"All plots saved:\n"
      f" • county_level_predictions_h{horizon}.pdf\n"
      f" • loss_over_epochs.pdf\n"
      f" • national_level_test.pdf")
