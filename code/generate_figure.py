#!/usr/bin/env python3
"""
Generate confusion matrix figure for the paper.
Uses saved results from chemistry_comparison.json

IMPORTANT: Values are read from saved data, not hardcoded.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load results
with open("results/chemistry_comparison.json", "r") as f:
    data = json.load(f)

# Extract confusion matrices and metrics
conf_standard = np.array(data['standard']['confusion'])
conf_messy = np.array(data['messy']['confusion'])

# Extract metrics from data (not hardcoded)
std_d_eff = data['standard']['d_eff']
std_accuracy = data['standard']['accuracy'] * 100
std_species = data['standard']['n_species']

messy_d_eff = data['messy']['d_eff']
messy_accuracy = data['messy']['accuracy'] * 100
messy_species = data['messy']['n_species']

print(f"Standard: {std_species} species, D_eff={std_d_eff:.2f}, Acc={std_accuracy:.0f}%")
print(f"Messy: {messy_species} species, D_eff={messy_d_eff:.2f}, Acc={messy_accuracy:.0f}%")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Standard chemistry (15 species)
ax1 = axes[0]
im1 = ax1.imshow(conf_standard, cmap='Blues', aspect='equal')
ax1.set_xlabel('Predicted Environment', fontsize=11)
ax1.set_ylabel('True Environment', fontsize=11)
ax1.set_title(f'(A) {std_species} species\n$D_{{eff}}$ = {std_d_eff:.2f}, Accuracy = {std_accuracy:.0f}%',
              fontsize=12, fontweight='bold')
ax1.set_xticks([0, 7, 15, 23, 31])
ax1.set_yticks([0, 7, 15, 23, 31])

# Panel B: High-species chemistry (50 species)
ax2 = axes[1]
im2 = ax2.imshow(conf_messy, cmap='Reds', aspect='equal')
ax2.set_xlabel('Predicted Environment', fontsize=11)
ax2.set_ylabel('True Environment', fontsize=11)
ax2.set_title(f'(B) {messy_species} species\n$D_{{eff}}$ = {messy_d_eff:.2f}, Accuracy = {messy_accuracy:.0f}%',
              fontsize=12, fontweight='bold')
ax2.set_xticks([0, 7, 15, 23, 31])
ax2.set_yticks([0, 7, 15, 23, 31])

# Add colorbars
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Count', fontsize=10)
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Count', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved figures/fig_confusion_matrix.pdf")
print("Saved figures/fig_confusion_matrix.png")

# Also generate timescale comparison if available
try:
    with open("results/timescale_comparison.json", "r") as f:
        ts_data = json.load(f)

    # Create a simple bar chart for timescale comparison
    fig2, ax = plt.subplots(figsize=(8, 5))

    metrics = ['Unique Codes\n(/32)', '$D_{eff}$', 'Accuracy\n(%)']
    fast_vals = [ts_data['fast']['unique_codes'],
                 ts_data['fast']['d_eff'],
                 ts_data['fast']['accuracy'] * 100]
    mixed_vals = [ts_data['mixed']['unique_codes'],
                  ts_data['mixed']['d_eff'],
                  ts_data['mixed']['accuracy'] * 100]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, fast_vals, width, label='All-fast', color='#1f77b4')
    bars2 = ax.bar(x + width/2, mixed_vals, width, label='Mixed (30% slow)', color='#2ca02c')

    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Timescale Separation Improves Code Emergence', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars1, fast_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}' if val < 10 else f'{val:.0f}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, mixed_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}' if val < 10 else f'{val:.0f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/fig_timescale.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_timescale.png', dpi=300, bbox_inches='tight')
    print("Saved figures/fig_timescale.pdf")
    print("Saved figures/fig_timescale.png")

except FileNotFoundError:
    print("No timescale comparison data found, skipping that figure")

print("\nDone!")
