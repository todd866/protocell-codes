#!/usr/bin/env python3
"""
Figure 1: Code Emergence Architecture
Two-row layout for clarity
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], wspace=0.3, hspace=0.35)

# =============================================================================
# Panel A: Hexagonal vesicle array (top-left, larger)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

def hex_grid(n_rings):
    coords = [(0, 0)]
    for ring in range(1, n_rings + 1):
        for i in range(6):
            angle = i * np.pi / 3
            for j in range(ring):
                x = ring * np.cos(angle) - j * np.cos(angle + np.pi/3)
                y = ring * np.sin(angle) - j * np.sin(angle + np.pi/3)
                coords.append((x * 0.9, y * 0.9))
    return np.array(coords)

coords = hex_grid(4)  # 61 vesicles

# Draw coupling lines first (behind vesicles)
for i, (x1, y1) in enumerate(coords):
    for j, (x2, y2) in enumerate(coords):
        if i < j:
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < 1.2:
                ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.2, linewidth=0.5, zorder=1)

# Draw vesicles
for i, (x, y) in enumerate(coords):
    dist = np.sqrt(x**2 + y**2)
    color = plt.cm.Blues(0.3 + 0.5 * dist / 4)
    hex_patch = RegularPolygon((x, y), numVertices=6, radius=0.4,
                                facecolor=color, edgecolor='#333333',
                                linewidth=0.6, alpha=0.85, zorder=2)
    ax1.add_patch(hex_patch)

ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(-4.5, 4.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left', x=-0.05)
ax1.text(0, -5.2, '61 coupled vesicles\n128D internal state each\nWeak neighbor coupling',
         ha='center', fontsize=9, color='#444444')

# =============================================================================
# Panel B: Substrate competition (top-right)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

x = np.linspace(0, 1, 100)

# Raw activations
a1_raw = np.exp(-((x - 0.25)**2) / 0.015) * 0.7
a2_raw = np.exp(-((x - 0.50)**2) / 0.015) * 0.5
a3_raw = np.exp(-((x - 0.75)**2) / 0.015) * 0.85

# After competition
h = 4
epsilon = 0.01
total = a1_raw**h + a2_raw**h + a3_raw**h + epsilon
a1_comp = a1_raw**h / total
a2_comp = a2_raw**h / total
a3_comp = a3_raw**h / total

ax2.fill_between(x, 0, a1_comp, alpha=0.7, color='#e74c3c', label='Channel 1')
ax2.fill_between(x, a1_comp, a1_comp + a2_comp, alpha=0.7, color='#3498db', label='Channel 2')
ax2.fill_between(x, a1_comp + a2_comp, a1_comp + a2_comp + a3_comp, alpha=0.7, color='#2ecc71', label='Channel 3')

ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax2.text(0.5, 0.53, 'winner threshold', fontsize=8, color='gray', ha='center', va='bottom',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

ax2.set_xlabel('Environmental gradient', fontsize=10)
ax2.set_ylabel('Channel activation', fontsize=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.05)
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left', x=-0.12)
ax2.legend(fontsize=8, loc='upper right', framealpha=0.95)

# Equation inside panel
ax2.text(0.02, 0.02, r'$\phi_j = \frac{a_j^h}{\sum_k a_k^h + \epsilon}$',
         transform=ax2.transAxes, fontsize=9, ha='left', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=2))

# =============================================================================
# Panel C: Input-output mapping (bottom, full width)
# =============================================================================
ax3 = fig.add_subplot(gs[1, :])

n_examples = 8
env_labels = ['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111']
# Channel indices (0-29) as actual codes - each is a 4-symbol sequence
codes = [[3, 17, 8, 22], [5, 12, 26, 1], [9, 0, 19, 14], [21, 7, 3, 28],
         [15, 24, 11, 6], [2, 18, 29, 10], [27, 4, 16, 23], [13, 20, 5, 8]]

for i, (env, code) in enumerate(zip(env_labels, codes)):
    x_start = i * 0.9
    y = 0.5

    # Environment bits
    for j, bit in enumerate(env):
        color = '#2c3e50' if bit == '1' else '#ecf0f1'
        rect = plt.Rectangle((x_start + j*0.12, y + 0.1), 0.1, 0.25,
                              facecolor=color, edgecolor='#7f8c8d', linewidth=0.8)
        ax3.add_patch(rect)

    # Arrow
    ax3.annotate('', xy=(x_start + 0.35, y - 0.05), xytext=(x_start + 0.35, y + 0.08),
                 arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2))

    # Code symbols (channel indices)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for j, (chan, col) in enumerate(zip(code, colors)):
        ax3.text(x_start + 0.08 + j*0.18, y - 0.25, str(chan), fontsize=9, fontweight='bold',
                 color=col, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=col, linewidth=1))

ax3.set_xlim(-0.2, 7.4)
ax3.set_ylim(-0.1, 1.1)
ax3.axis('off')
ax3.set_title('C', fontsize=12, fontweight='bold', loc='left', x=-0.02, y=0.85)

# Labels
ax3.text(3.5, 1.0, 'Environment (5-bit) → 4-channel code (indices 0-29)', fontsize=11, ha='center', fontweight='bold')
ax3.text(3.5, -0.05, '32 unique mappings  •  98.4% reproducibility  •  335,000× separation',
         fontsize=10, ha='center', color='#555555')

plt.savefig('fig1_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig1_architecture.png', dpi=300, bbox_inches='tight')
print("Saved fig1_architecture.pdf/png")
