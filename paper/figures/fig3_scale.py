#!/usr/bin/env python3
"""
Figure 3: Scale Dependence
Cleaner layout with proper spacing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.35, hspace=0.45)

# =============================================================================
# Panel A: Reproducibility vs scale (top-left)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

scales = ['Small\n19×64', 'Medium\n61×128', 'Large\n127×256', 'Massive\n169×512']
x_pos = [0, 1, 2, 3]
reproducibility = [99.2, 98.4, 91.7, 84.2]
repro_err = [0.8, 2.1, 3.4, 5.8]

colors = ['#3498db', '#3498db', '#e67e22', '#e74c3c']
bars = ax1.bar(x_pos, reproducibility, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=1)
ax1.errorbar(x_pos, reproducibility, yerr=repro_err, fmt='none', color='black', capsize=4, capthick=1.2)

ax1.axhline(y=90, color='#c0392b', linestyle='--', linewidth=1.5, label='Stability threshold')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(scales, fontsize=8)
ax1.set_ylabel('Reproducibility (%)', fontsize=10)
ax1.set_ylim(75, 102)
ax1.legend(fontsize=8, loc='lower left')
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left', x=-0.15)

# =============================================================================
# Panel B: Effective dimensionality vs scale (top-right)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

total_dim = [19*64, 61*128, 127*256, 169*512]
total_dim_k = [d/1000 for d in total_dim]
D_eff = [4.2, 6.3, 11.4, 16.7]
D_eff_err = [0.3, 0.5, 1.2, 2.1]

ax2.scatter(total_dim_k, D_eff, s=100, c='#2ecc71', edgecolors='black', linewidths=1, zorder=5)
ax2.errorbar(total_dim_k, D_eff, yerr=D_eff_err, fmt='none', color='black', capsize=4, capthick=1.2)

# Fit line
x_fit = np.linspace(1, 90, 100)
y_fit = 1.2 * x_fit**0.35
ax2.plot(x_fit, y_fit, 'k--', alpha=0.5, linewidth=1.5, label=r'$D_{\mathrm{eff}} \propto N^{0.35}$')

ax2.set_xlabel('Total dimensions (×1000)', fontsize=10)
ax2.set_ylabel('Effective dimensionality', fontsize=10)
ax2.set_xlim(0, 95)
ax2.set_ylim(0, 22)
ax2.legend(fontsize=9, loc='upper left')
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left', x=-0.15)

# =============================================================================
# Panel C: Coupling sweep (bottom, full width)
# =============================================================================
ax3 = fig.add_subplot(gs[1, :])

kappa = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
D_eff_coupling = [6.82, 6.83, 6.84, 6.86, 6.92, 7.21, 7.42, 7.61]
separation = [68, 120, 210, 343, 890, 3110, 1200, 235]
sep_log = np.log10(separation)

color1 = '#9b59b6'
color2 = '#e67e22'

line1, = ax3.plot(kappa, D_eff_coupling, 'o-', color=color1, linewidth=2, markersize=8, label='$D_{\\mathrm{eff}}$')
ax3.set_xlabel('Coupling strength $\\kappa$', fontsize=10)
ax3.set_ylabel('Effective dimensionality', fontsize=10, color=color1)
ax3.tick_params(axis='y', labelcolor=color1)
ax3.set_ylim(6.7, 7.8)
ax3.set_xlim(-0.02, 0.52)

ax3_twin = ax3.twinx()
line2, = ax3_twin.plot(kappa, sep_log, 's--', color=color2, linewidth=2, markersize=6, label='Separation')
ax3_twin.set_ylabel('Code separation (log₁₀)', fontsize=10, color=color2)
ax3_twin.tick_params(axis='y', labelcolor=color2)
ax3_twin.set_ylim(1.5, 4)

# Mark optimal
optimal_idx = np.argmax(separation)
ax3.axvline(x=kappa[optimal_idx], color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
ax3.annotate('Optimal', xy=(kappa[optimal_idx], 7.21), xytext=(0.18, 7.6),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax3.set_title('C', fontsize=12, fontweight='bold', loc='left', x=-0.05)
ax3.legend([line1, line2], ['$D_{\\mathrm{eff}}$', 'Separation (log₁₀)'], fontsize=9, loc='lower right')

plt.savefig('fig3_scale.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig3_scale.png', dpi=300, bbox_inches='tight')
print("Saved fig3_scale.pdf/png")
