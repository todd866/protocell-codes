#!/usr/bin/env python3
"""
Generate Figure: Experimental Validation in Non-Living Systems

Three-panel figure showing predator-prey dynamics emerge from physics:
A) Dusty plasma Lotka-Volterra oscillations (Ross & McKenzie 2016)
B) Chemotactic droplet chase trajectories (Meredith et al. 2020)
C) Mode-switching coherence transitions (Horibe et al. 2011)

Author: Ian Todd
Date: January 2026
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Set style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# =============================================================================
# Panel A: Dusty Plasma Lotka-Volterra
# =============================================================================

def dusty_plasma_ode(t, y, params):
    x1, x2 = y
    x1 = max(x1, 1e-10)
    x2 = max(x2, 1e-10)

    dx1 = x1 * (params['alpha'] - params['beta'] * x2) - params['k'] * x1**params['n']
    dx2 = x2 * (-params['gamma'] + params['delta'] * x1)

    return [dx1, dx2]


def run_dusty_plasma(n_value):
    params = {
        'alpha': 1.5,
        'beta': 0.02,
        'gamma': 0.8,
        'delta': 0.01,
        'k': 0.0002,
        'n': n_value,
    }
    y0 = [80.0, 30.0]
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 2000)

    sol = solve_ivp(dusty_plasma_ode, t_span, y0, args=(params,),
                    t_eval=t_eval, method='RK45', max_step=0.1)
    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# Panel B: Chemotactic Droplet Chase
# =============================================================================

def run_chemotactic_chase(seed=42):
    rng = default_rng(seed)

    params = {
        'k_attract': 0.5,
        'k_repel': 0.3,
        'range': 5.0,
        'noise': 0.1,
        'capture_radius': 0.5,
    }

    # Initial positions
    x_pred, y_pred = -4.0, 0.0
    x_prey, y_prey = 4.0, 1.0

    trajectory = {
        'pred': [(x_pred, y_pred)],
        'prey': [(x_prey, y_prey)],
    }

    dt = 0.1
    for step in range(500):
        dx = x_prey - x_pred
        dy = y_prey - y_pred
        r = np.sqrt(dx**2 + dy**2) + 1e-6
        ux, uy = dx/r, dy/r

        # Predator attracted
        vx_pred = params['k_attract'] * ux / (1 + r/params['range'])
        vy_pred = params['k_attract'] * uy / (1 + r/params['range'])

        # Prey repelled + noise
        vx_prey = -params['k_repel'] * ux / (1 + r/params['range']) + rng.normal(0, params['noise'])
        vy_prey = -params['k_repel'] * uy / (1 + r/params['range']) + rng.normal(0, params['noise'])

        x_pred += vx_pred * dt
        y_pred += vy_pred * dt
        x_prey += vx_prey * dt
        y_prey += vy_prey * dt

        trajectory['pred'].append((x_pred, y_pred))
        trajectory['prey'].append((x_prey, y_prey))

        if r < params['capture_radius']:
            break

    return trajectory


# =============================================================================
# Panel C: Mode Switching
# =============================================================================

def run_mode_switching(seed=42):
    rng = default_rng(seed)

    T, dt = 300, 0.5
    C = 0.5  # Start at middle coherence
    thresholds = [0.25, 0.5, 0.75]

    history = {'t': [0], 'C': [C], 'mode': [1]}

    for step in range(int(T/dt)):
        t = step * dt

        # Coherence dynamics with noise
        dC = 0.1 * (0.5 - C) + rng.normal(0, 0.08)
        C = np.clip(C + dC * dt, 0.01, 0.99)

        # Determine mode
        mode = 0
        if C > thresholds[0]: mode = 1
        if C > thresholds[1]: mode = 2
        if C > thresholds[2]: mode = 3

        history['t'].append(t)
        history['C'].append(C)
        history['mode'].append(mode)

    return history


# =============================================================================
# Generate Figure
# =============================================================================

def main():
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Panel A: Dusty Plasma
    ax_a = fig.add_subplot(gs[0, 0])

    t, x1, x2 = run_dusty_plasma(n_value=2.0)

    ax_a.plot(t, x1/x1.max(), 'b-', alpha=0.8, lw=1.5, label='Prey (electrons)')
    ax_a.plot(t, x2/x2.max(), 'r-', alpha=0.8, lw=1.5, label='Predator (dust)')
    ax_a.set_xlabel('Time')
    ax_a.set_ylabel('Normalized Population')
    ax_a.set_title('A. Dusty Plasma (Ross & McKenzie 2016)')
    ax_a.legend(loc='upper right', framealpha=0.9)
    ax_a.set_xlim(0, 200)
    ax_a.set_ylim(0, 1.1)

    # Add annotation about n=2
    ax_a.text(0.05, 0.95, 'n = 2 (stable)', transform=ax_a.transAxes,
              fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Chemotactic Chase
    ax_b = fig.add_subplot(gs[0, 1])

    traj = run_chemotactic_chase()
    pred = np.array(traj['pred'])
    prey = np.array(traj['prey'])

    # Color by time
    n_points = len(pred)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    for i in range(n_points - 1):
        ax_b.plot(pred[i:i+2, 0], pred[i:i+2, 1], c='red', alpha=0.5, lw=1)
        ax_b.plot(prey[i:i+2, 0], prey[i:i+2, 1], c='blue', alpha=0.5, lw=1)

    # Start and end markers
    ax_b.scatter(pred[0, 0], pred[0, 1], c='darkred', s=100, marker='o', zorder=5, label='Predator start')
    ax_b.scatter(prey[0, 0], prey[0, 1], c='darkblue', s=100, marker='s', zorder=5, label='Prey start')
    ax_b.scatter(pred[-1, 0], pred[-1, 1], c='red', s=150, marker='*', zorder=5)
    ax_b.scatter(prey[-1, 0], prey[-1, 1], c='blue', s=150, marker='*', zorder=5)

    ax_b.set_xlabel('x position')
    ax_b.set_ylabel('y position')
    ax_b.set_title('B. Chemotactic Chase (Meredith 2020)')
    ax_b.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax_b.set_aspect('equal')
    ax_b.set_xlim(-6, 8)
    ax_b.set_ylim(-4, 6)

    # Add capture annotation
    ax_b.annotate('Capture', xy=(pred[-1, 0], pred[-1, 1]), xytext=(pred[-1, 0]+1.5, pred[-1, 1]+1),
                  fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    # Panel C: Mode Switching
    ax_c = fig.add_subplot(gs[0, 2])

    history = run_mode_switching()
    t = np.array(history['t'])
    C = np.array(history['C'])
    modes = np.array(history['mode'])

    # Plot coherence
    ax_c.plot(t, C, 'k-', alpha=0.8, lw=1.5, label='Coherence')

    # Add threshold lines (labels inside plot area)
    for thresh, name in zip([0.25, 0.5, 0.75], ['Fluct/Vib', 'Vib/Dir', 'Dir/Circ']):
        ax_c.axhline(thresh, color='gray', linestyle='--', alpha=0.5, lw=1)
        ax_c.text(290, thresh + 0.03, name, fontsize=7, va='bottom', ha='right')

    # Color background by mode
    mode_colors = ['lightgray', 'lightblue', 'lightgreen', 'lightsalmon']
    mode_names = ['Fluctuating', 'Vibrational', 'Directional', 'Circular']

    for i in range(len(t) - 1):
        ax_c.axvspan(t[i], t[i+1], alpha=0.3, color=mode_colors[modes[i]], lw=0)

    ax_c.set_xlabel('Time')
    ax_c.set_ylabel('Coherence C')
    ax_c.set_title('C. Mode-Switching (Horibe 2011)')
    ax_c.set_xlim(0, 300)
    ax_c.set_ylim(0, 1)

    # Legend for modes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=mode_colors[i], alpha=0.5, label=mode_names[i])
                       for i in range(4)]
    ax_c.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

    plt.tight_layout()

    # Save
    plt.savefig('fig_experimental_validation.pdf', bbox_inches='tight')
    plt.savefig('fig_experimental_validation.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_experimental_validation.pdf/png")

    plt.close()


if __name__ == '__main__':
    main()
