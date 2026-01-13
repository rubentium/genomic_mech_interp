"""
Visualization Module for Mechanistic Interpretability Analysis

This module contains all plotting and visualization functions for:
- Attention heatmaps
- Indirect effect analysis
- Head-level circuit analysis
- Logit comparisons
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    layer_idx: int,
    head_idx: Optional[int] = None,
    mutation_position: Optional[int] = None,
    title_suffix: str = "",
    file_prefix: str = "",
) -> None:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention tensor [batch, num_heads, seq_len, seq_len]
        layer_idx: Which layer these attention weights are from
        head_idx: Specific attention head to visualize. If None, average across heads.
        mutation_position: Highlight this position (e.g., where mutation occurred)
        title_suffix: Additional text for the title
        file_prefix: Prefix for output filename (e.g., 'benign_' or 'malignant_')
    """
    # attention_weights shape: [batch, num_heads, seq_len, seq_len]
    attn = attention_weights[0].cpu().numpy()  # Remove batch dimension
    
    if head_idx is not None:
        attn = attn[head_idx]  # Select specific head
        head_label = f" Head {head_idx}"
    else:
        attn = attn.mean(axis=0)  # Average across all heads
        head_label = " (Averaged)"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap='viridis', square=True, cbar_kws={'label': 'Attention'})
    
    if mutation_position is not None:
        plt.axvline(x=mutation_position, color='red', linewidth=2, linestyle='--', label='Mutation')
        plt.axhline(y=mutation_position, color='red', linewidth=2, linestyle='--')
        plt.legend()
    
    plt.title(f"Layer {layer_idx}{head_label} Attention Weights{title_suffix}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/{file_prefix}attention_layer_{layer_idx}.png' if file_prefix else f'results/attention_layer_{layer_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_indirect_effects(
    layer_effects: Dict[int, float],
    highlight_layer: Optional[int] = None,
    file_prefix: str = "",
) -> None:
    """
    Plot the indirect effect of each layer as a bar chart.
    
    This visualization shows which layers contribute most to the difference
    between wild-type and mutant predictions.
    
    Args:
        layer_effects: Dictionary mapping layer index to indirect effect
        highlight_layer: Optionally highlight a specific layer
        file_prefix: Prefix for output filename (e.g., 'benign_' or 'malignant_')
    """
    layers = sorted(layer_effects.keys())
    effects = [layer_effects[l] for l in layers]
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if i == highlight_layer else 'steelblue' for i in layers]
    bars = plt.bar(layers, effects, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Indirect Effect (Prediction Flip Magnitude)", fontsize=12)
    plt.title("Mechanistic Interpretability: Indirect Effect by Layer", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/{file_prefix}indirect_effects_summary.png' if file_prefix else 'results/indirect_effects_summary.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_head_effects(
    head_effects: Dict[int, float],
    layer_idx: int,
    highlight_heads: Optional[List[int]] = None,
    file_prefix: str = "",
) -> None:
    """
    Plot the indirect effect of each attention head within a layer.
    
    This circuit analysis visualization shows which specific heads are driving
    the model's mutation detection.
    
    Args:
        head_effects: Dictionary mapping head index to indirect effect
        layer_idx: Which layer these heads belong to
        highlight_heads: List of head indices to highlight
        file_prefix: Prefix for output filename
    """
    heads = sorted(head_effects.keys())
    effects = [head_effects[h] for h in heads]
    
    plt.figure(figsize=(14, 7))
    
    # Color code: highlight specified heads
    if highlight_heads is None:
        # Auto-highlight top 3 heads by absolute effect
        sorted_by_effect = sorted(head_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        highlight_heads = [h for h, _ in sorted_by_effect[:3]]
    
    colors = ['red' if i in highlight_heads else 'steelblue' for i in heads]
    bars = plt.bar(heads, effects, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=8,
        )
    
    plt.xlabel("Attention Head Index", fontsize=12)
    plt.ylabel("Indirect Effect (Contribution to Prediction)", fontsize=12)
    plt.title(f"Circuit Analysis: Attention Head Effects in Layer {layer_idx}", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend for highlighted heads
    if highlight_heads:
        plt.legend(
            [f'Critical Heads: {highlight_heads}'],
            loc='upper right',
            fontsize=10
        )
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/{file_prefix}head_effects_layer_{layer_idx}.png' if file_prefix else f'results/head_effects_layer_{layer_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHead analysis visualization saved: {filename}")


def compare_logits_by_position(
    original_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    positions: Optional[List[int]] = None,
    file_prefix: str = "",
) -> None:
    """
    Compare logits at specific positions across original, patched, and corrupt sequences.
    
    Args:
        original_logits: Logits from clean sequence [1, seq_len, vocab_size]
        patched_logits: Logits from patched clean sequence [1, seq_len, vocab_size]
        corrupt_logits: Logits from corrupt sequence [1, seq_len, vocab_size]
        positions: Which positions to visualize. If None, samples evenly across sequence.
        file_prefix: Prefix for output filename (e.g., 'benign_' or 'malignant_')
    """
    max_logit_orig = original_logits.max(dim=-1)[0][0].cpu().numpy()
    max_logit_patched = patched_logits.max(dim=-1)[0][0].cpu().numpy()
    max_logit_corrupt = corrupt_logits.max(dim=-1)[0][0].cpu().numpy()
    
    # Handle different sequence lengths
    min_len = min(len(max_logit_orig), len(max_logit_patched), len(max_logit_corrupt))
    max_logit_orig = max_logit_orig[:min_len]
    max_logit_patched = max_logit_patched[:min_len]
    max_logit_corrupt = max_logit_corrupt[:min_len]
    
    if positions is None:
        seq_len = min_len
        positions = np.linspace(0, seq_len - 1, min(10, seq_len), dtype=int).tolist()
    
    # Filter positions to be within bounds
    if positions is not None:
        positions = [p for p in positions if p < min_len]
    else:
        positions = []
    
    x = np.arange(len(positions))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, [max_logit_orig[p] for p in positions], width, label='Original (Clean)', alpha=0.8)
    plt.bar(x, [max_logit_patched[p] for p in positions], width, label='Patched', alpha=0.8)
    plt.bar(x + width, [max_logit_corrupt[p] for p in positions], width, label='Corrupt', alpha=0.8)
    
    plt.xlabel("Sequence Position", fontsize=12)
    plt.ylabel("Maximum Logit", fontsize=12)
    plt.title("Logit Comparison: Original vs Patched vs Corrupt", fontsize=14)
    plt.xticks(x, [str(p) for p in positions])  # Convert to strings
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/{file_prefix}logits_comparison.png' if file_prefix else 'results/logits_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
