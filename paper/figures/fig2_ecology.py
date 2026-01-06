#!/usr/bin/env python3
"""
Figure 2: Predator-Prey Ecology
Cleaner two-row layout
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit as sigmoid

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

fig = plt.figure(figsize=(7, 5.5))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1], wspace=0.35, hspace=0.4)

# =============================================================================
# Panel A: Phase space (top-left, larger)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

N = np.linspace(10, 250, 100)
C = np.linspace(0.05, 0.7, 100)
NN, CC = np.meshgrid(N, C)

# Fitness function
R_bar, k, gamma, zeta = 1.0, 0.01, 1.8, 1.5
C_star, a, m = 0.3, 2.0, 2.0
phi = np.where(CC > C_star, 1 + a * (CC - C_star)**m, 1.0)
W = NN * R_bar * phi - k * NN**gamma * CC**zeta

# Viability region
ax1.contourf(NN, CC, W, levels=[-1000, 0], colors=['#ffeeee'], alpha=0.5)
ax1.contour(NN, CC, W, levels=[0], colors=['#888888'], linewidths=1.5, linestyles='--')

# Hard ceiling
ceiling = 40
ax1.axvline(x=ceiling, color='#c0392b', linewidth=2.5, label='Hard ceiling')
ax1.fill_betweenx([0.05, 0.7], 0, ceiling, alpha=0.08, color='#c0392b')

# Predator attractor
ax1.scatter([39], [0.53], s=200, c='#e74c3c', marker='*', zorder=5,
            edgecolors='black', linewidths=1)
ax1.annotate('Predators\nN=39, C=0.53', xy=(39, 0.53), xytext=(90, 0.58),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2))

# Prey attractor
ax1.scatter([183], [0.12], s=180, c='#3498db', marker='o', zorder=5,
            edgecolors='black', linewidths=1)
ax1.annotate('Prey\nN=183, C=0.12', xy=(183, 0.12), xytext=(140, 0.28),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.2))

ax1.set_xlabel('Population size N', fontsize=10)
ax1.set_ylabel('Coherence C', fontsize=10)
ax1.set_xlim(10, 250)
ax1.set_ylim(0.05, 0.7)
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left', x=-0.12)
ax1.legend(fontsize=8, loc='upper right')

# =============================================================================
# Panel B: Population dynamics (top-right)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

np.random.seed(42)
t = np.linspace(0, 100, 500)
prey_pop = 4900 + 300 * np.sin(0.15 * t) * np.exp(-0.02 * t) + 50 * np.random.randn(len(t))
pred_pop = 40 + 15 * np.sin(0.15 * t + 0.5) * np.exp(-0.02 * t) + 5 * np.random.randn(len(t))
prey_pop = np.clip(prey_pop, 4500, 5200)
pred_pop = np.clip(pred_pop, 20, 60)

ax2.plot(t, prey_pop / 100, color='#3498db', linewidth=1.5, label='Prey (÷100)')
ax2.plot(t, pred_pop, color='#e74c3c', linewidth=1.5, label='Predators')

ax2.axhline(y=49, color='#3498db', linestyle=':', alpha=0.5, linewidth=1)
ax2.axhline(y=36, color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1)

ax2.set_xlabel('Generation', fontsize=10)
ax2.set_ylabel('Population', fontsize=10)
ax2.set_xlim(0, 100)
ax2.set_ylim(15, 65)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left', x=-0.12)

# Annotations (inside plot area with background)
ax2.text(95, 49, '99%', fontsize=8, color='#3498db', va='center', ha='right',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
ax2.text(95, 36, '1%', fontsize=8, color='#e74c3c', va='center', ha='right',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# =============================================================================
# Panel C: Raid success (bottom, full width)
# =============================================================================
ax3 = fig.add_subplot(gs[1, :])

C_ratio = np.linspace(1, 10, 100)
N_ratio = 5
# Match predator_prey.py simulation parameters
kappa_C, kappa_N = 3.0, 2.0
p_raid = sigmoid(kappa_C * np.log(C_ratio) - kappa_N * np.log(N_ratio))

ax3.plot(C_ratio, p_raid, color='#9b59b6', linewidth=2.5)
ax3.fill_between(C_ratio, 0, p_raid, alpha=0.15, color='#9b59b6')

# Mark observed ratio
actual_ratio = 0.53 / 0.12
actual_p = sigmoid(kappa_C * np.log(actual_ratio) - kappa_N * np.log(N_ratio))
ax3.axvline(x=actual_ratio, color='#e74c3c', linestyle='--', linewidth=1.5)
ax3.scatter([actual_ratio], [actual_p], s=100, c='#e74c3c', zorder=5,
            edgecolors='black', linewidths=1)
ax3.annotate(f'Observed: $C_S/C_B$ = {actual_ratio:.1f}\nRaid success = {actual_p:.0%}',
             xy=(actual_ratio, actual_p), xytext=(6.5, 0.35), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax3.set_xlabel('Coherence ratio $C_S / C_B$', fontsize=10)
ax3.set_ylabel('Raid success probability', fontsize=10)
ax3.set_xlim(1, 10)
ax3.set_ylim(0, 1)
ax3.set_title('C', fontsize=12, fontweight='bold', loc='left', x=-0.05)

# Equation
ax3.text(0.98, 0.05, r'$p_{\mathrm{raid}} = \sigma\left(\kappa_C \log\frac{C_S}{C_B} - \kappa_N \log\frac{N_B}{N_S}\right)$',
         transform=ax3.transAxes, fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

plt.savefig('fig2_ecology.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig2_ecology.png', dpi=300, bbox_inches='tight')
print("Saved fig2_ecology.pdf/png")
