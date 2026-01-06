#!/usr/bin/env python3
"""
Simple ablation figure for HeroX paper.
Shows: h=1 vs h=4 discretization effect
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate substrate competition outputs at different h values
def softmax_competition(activations, h=4):
    """Substrate competition with Hill coefficient h"""
    powered = activations ** h
    return powered / (powered.sum() + 1e-10)

# Generate random activations (simulating reservoir output)
n_trials = 1000
n_channels = 8

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Panel A: Distribution of max output for h=1 vs h=4
h_values = [1, 2, 4]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, h in enumerate(h_values):
    max_outputs = []
    for _ in range(n_trials):
        # Random activations with some structure
        activations = np.random.exponential(1, n_channels)
        outputs = softmax_competition(activations, h=h)
        max_outputs.append(outputs.max())

    axes[0].hist(max_outputs, bins=30, alpha=0.6, label=f'h={h}', color=colors[idx])

axes[0].set_xlabel('Max output value')
axes[0].set_ylabel('Count')
axes[0].set_title('A. Discretization strength')
axes[0].legend()
axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.5)

# Panel B: Winner margin distribution
for idx, h in enumerate(h_values):
    margins = []
    for _ in range(n_trials):
        activations = np.random.exponential(1, n_channels)
        outputs = softmax_competition(activations, h=h)
        sorted_out = np.sort(outputs)[::-1]
        margin = sorted_out[0] / (sorted_out[1] + 1e-10)
        margins.append(min(margin, 20))  # Cap for visualization

    axes[1].hist(margins, bins=30, alpha=0.6, label=f'h={h}', color=colors[idx])

axes[1].set_xlabel('Winner margin (1st / 2nd)')
axes[1].set_ylabel('Count')
axes[1].set_title('B. Winner margin amplification')
axes[1].legend()
axes[1].axvline(2, color='gray', linestyle='--', alpha=0.5, label='2x threshold')

# Panel C: Accuracy vs noise at different h
noise_levels = np.linspace(0, 0.5, 20)
h_test = [1, 2, 4, 8]

for h in h_test:
    accuracies = []
    for noise in noise_levels:
        correct = 0
        for _ in range(200):
            # True signal
            activations = np.random.exponential(1, n_channels)
            true_winner = np.argmax(softmax_competition(activations, h=h))

            # Noisy signal
            noisy = activations + noise * np.random.randn(n_channels)
            noisy = np.maximum(noisy, 0.01)
            noisy_winner = np.argmax(softmax_competition(noisy, h=h))

            if true_winner == noisy_winner:
                correct += 1
        accuracies.append(correct / 200)

    axes[2].plot(noise_levels * 100, np.array(accuracies) * 100,
                 label=f'h={h}', linewidth=2)

axes[2].set_xlabel('Noise level (%)')
axes[2].set_ylabel('Winner consistency (%)')
axes[2].set_title('C. Noise robustness')
axes[2].legend()
axes[2].set_ylim([0, 105])
axes[2].axhline(95, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig4_ablation_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('fig4_ablation_analysis.pdf', bbox_inches='tight')
print("Saved fig4_ablation_analysis.png/pdf")

# Summary stats
print("\nAblation Summary:")
print(f"h=1: Mean max output = {np.mean([softmax_competition(np.random.exponential(1,8), h=1).max() for _ in range(100)]):.2f}")
print(f"h=4: Mean max output = {np.mean([softmax_competition(np.random.exponential(1,8), h=4).max() for _ in range(100)]):.2f}")
