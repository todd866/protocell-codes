#!/usr/bin/env python3
"""
Confusion Matrix Visualization
==============================

Generates confusion matrix plot for decoder accuracy.

Usage:
    python visualize_confusion.py [--scale medium] [--trials 5]

Author: Ian Todd
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import code_emergence_core as core


def plot_confusion_matrix(cm: np.ndarray, title: str = "Decoder Confusion Matrix",
                          save_path: str = None):
    """Plot confusion matrix with heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize by row (true class)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Classification Rate')

    # Labels
    ax.set_xlabel('Predicted Environment')
    ax.set_ylabel('True Environment')
    ax.set_title(title)

    # Ticks
    tick_positions = [0, 7, 15, 23, 31]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    # Add diagonal accuracy text
    diagonal_acc = np.diag(cm).sum() / cm.sum()
    ax.text(0.02, 0.98, f'Overall Accuracy: {100*diagonal_acc:.1f}%',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix")
    parser.add_argument("--scale", default="medium", help="Simulation scale")
    parser.add_argument("--trials", type=int, default=5, help="Trials per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", default=None, help="Save path for figure")
    args = parser.parse_args()

    core.configure(args.scale)
    print(f"Running {args.scale} scale ({core.N_VESICLES} vesicles)...")

    result = core.run_systematic_trials(n_trials=args.trials, seed=args.seed,
                                        return_confusion=True)

    if 'confusion_matrix' in result:
        cm = result['confusion_matrix']
        print(f"\nAccuracy: {100*result['decoding_accuracy']:.1f}%")
        print(f"Per-class accuracy: {100*result['per_class_accuracy_mean']:.1f}% mean")

        # Print error summary
        print("\nTop misclassifications:")
        errors = []
        for i in range(32):
            for j in range(32):
                if i != j and cm[i, j] > 0:
                    errors.append((cm[i, j], i, j))
        errors.sort(reverse=True)
        for count, true, pred in errors[:5]:
            print(f"  Config {true} → {pred}: {count} times")

        save_path = args.save or f"confusion_matrix_{args.scale}.pdf"
        plot_confusion_matrix(cm, title=f"Decoder Confusion Matrix ({args.scale} scale)",
                             save_path=save_path)
    else:
        print("No confusion matrix available")


if __name__ == "__main__":
    main()
