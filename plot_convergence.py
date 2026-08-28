import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
log_path = Path("experiments/logs/training_log.json")
output_path = Path("experiments/logs/convergence_curves.png")

if not log_path.exists():
    print(f"ERROR: Log file not found at {log_path}. Make sure training has completed first!")
    exit(1)

# Load training logs
with open(log_path, "r") as f:
    history = json.load(f)

rounds = [entry["round"] + 1 for entry in history]
ssl_losses = [entry["mean_ssl_loss"] for entry in history]

# Extract AUC metrics (only available every 5 rounds or final round)
auc_rounds = []
auc_scores = []
for entry in history:
    if "eval_metrics" in entry and "auc" in entry["eval_metrics"]:
        auc_rounds.append(entry["round"] + 1)
        auc_scores.append(entry["eval_metrics"]["auc"])

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Plot 1: Federated SSL Reconstruction Loss
ax1.plot(rounds, ssl_losses, color='#2c3e50', lw=2, marker='o', markersize=4, label='MAE Loss')
ax1.set_xlabel('Federated Communication Round', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_ylabel('Mean Reconstruction Loss (MSE)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_title('Self-Supervised Loss Convergence', fontsize=12, fontweight='bold', pad=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# Plot 2: Downstream Montgomery AUC
if auc_scores:
    ax2.plot(auc_rounds, auc_scores, color='#e74c3c', lw=2, marker='s', markersize=6, label='Montgomery AUC')
    ax2.set_xlabel('Federated Communication Round', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Classification AUC Score', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('Downstream TB Classification Performance', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylim([0.45, 1.02])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
else:
    ax2.text(0.5, 0.5, 'No evaluation metrics available yet\n(Computed every 5 rounds)', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=11, color='#7f8c8d')
    ax2.set_title('Downstream TB Classification Performance', fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()

# Save image
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
print(f"[OK] Convergence curves saved successfully to {output_path}")
