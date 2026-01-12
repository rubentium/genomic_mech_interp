"""
Main Execution Script for Mechanistic Interpretability Analysis

This script orchestrates the complete mechanistic interpretability pipeline
for analyzing how the Nucleotide Transformer v2 model processes DNA mutations.
"""

import os
import argparse
from typing import Dict, Optional

# Set custom HF cache directory BEFORE importing transformers
custom_cache_dir = os.path.expanduser("~/.cache/huggingface_custom")
os.environ["HF_HOME"] = custom_cache_dir
os.makedirs(custom_cache_dir, exist_ok=True)

from patching import (
    ModelLoader,
    NCBISequenceFetcher,
    run_inference,
    analyze_all_layers,
    analyze_heads_in_layer,
    patch_activations,
)
from plotter import (
    plot_attention_heatmap,
    plot_indirect_effects,
    plot_head_effects,
    compare_logits_by_position,
)


def main(
    use_ncbi: bool = True,
    use_tp53: bool = True,
    mutation_type: str = "malignant",
    run_head_analysis: bool = True
) -> tuple:
    """
    Main execution function demonstrating the complete pipeline.
    
    Args:
        use_ncbi: If True, fetch real genomic sequences from NCBI. If False, use synthetic test sequences.
        use_tp53: If True (and use_ncbi=True), fetch TP53 mutation. If False, use synthetic sequences.
        mutation_type: Type of TP53 mutation - 'malignant' (fetch_tp53_mutation) or 'benign' (fetch_tp53_benign_control)
        run_head_analysis: If True, perform detailed head-level circuit analysis on the critical layer
        
    Returns:
        Tuple of (model, tokenizer, device, layer_effects, head_effects)
    """
    
    print("=" * 70)
    print("Mechanistic Interpretability Analysis for Nucleotide Transformer v2")
    print("=" * 70)
    
    # Load model and tokenizer
    model, tokenizer, device = ModelLoader.load_model()
    
    # ========================================================================
    # Sequence Selection: NCBI Real Data vs Synthetic Test Sequences
    # ========================================================================
    
    if use_ncbi and use_tp53:
        print("\n" + "=" * 70)
        print(f"Using REAL genomic data from NCBI (TP53 {mutation_type.upper()} mutation)")
        print("=" * 70)
        
        try:
            if mutation_type.lower() == "benign":
                wild_type, mutant, mutation_pos = NCBISequenceFetcher.fetch_tp53_benign_control(flanking_bp=500)
            else:  # default to malignant
                wild_type, mutant, mutation_pos = NCBISequenceFetcher.fetch_tp53_mutation(flanking_bp=500)
        except Exception as e:
            print(f"\nFailed to fetch from NCBI: {e}")
            print("Falling back to synthetic sequences...\n")
            use_ncbi = False
    
    if not use_ncbi or not use_tp53:
        print("\n" + "=" * 70)
        print("Using SYNTHETIC test sequences")
        print("=" * 70)
        
        # Example sequences: wild-type and mutant
        # NOTE: Both sequences must be the SAME LENGTH to avoid tokenization issues
        # Create 128 bp sequences with a single nucleotide change
        base_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        wild_type = base_seq + base_seq  # 128 bp total
        
        # Create mutant with T->A substitution at position 63 (which has 'T')
        # Position 63 in the pattern ACGTACGT... is T (63 % 4 == 3)
        mutation_pos = 63
        mutant_list = list(wild_type)
        original_nucleotide = mutant_list[mutation_pos]
        mutant_list[mutation_pos] = "A"  # Change T to A
        mutant = "".join(mutant_list)
        
        print(f"\nWild-Type Sequence: {wild_type}")
        print(f"Mutant Sequence:    {mutant}")
        print(f"Sequence Length: {len(wild_type)} bp")
        print(f"Mutation: {original_nucleotide}->{mutant_list[mutation_pos]} at position {mutation_pos}")
    
    # Verify sequences are same length
    assert len(wild_type) == len(mutant), f"Sequences must be same length: {len(wild_type)} vs {len(mutant)}"
    
    print("Pre-capturing mutant activations for analysis...")
    corrupt_inf = run_inference(model, tokenizer, mutant, device)
    all_corrupt_states = corrupt_inf['hidden_states']
    
    # ========================================================================
    # Step 1: Analyze all layers to identify which contribute most
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Analyzing indirect effects across all layers...")
    print("=" * 70)
    
    # Use the mutation position we defined above
    print(f"Mutation position (character index): {mutation_pos}")
    
    layer_effects = analyze_all_layers(model, tokenizer, wild_type, mutant, mutation_pos, device)
    
    # Find the layer with maximum indirect effect
    max_effect_layer = max(layer_effects.keys(), key=lambda k: layer_effects[k])
    print(f"\nLayer with maximum indirect effect: Layer {max_effect_layer}")
    print(f"Maximum indirect effect: {layer_effects[max_effect_layer]:.4f}")
    
    # ========================================================================
    # Step 2: Detailed analysis of the most important layer
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 2: Detailed analysis of Layer {max_effect_layer}")
    print("=" * 70)
    
    detailed_result = patch_activations(
        model, 
        tokenizer, 
        wild_type, 
        mutant, 
        all_corrupt_states, 
        max_effect_layer, 
        mutation_pos, 
        device
    )
    
    # ========================================================================
    # Step 3: Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Generating visualizations...")
    print("=" * 70)
    
    # Create file prefix based on mutation type
    file_prefix = f"{mutation_type.lower()}_" if use_ncbi and use_tp53 else ""
    print(f"Output files will use prefix: '{file_prefix}'")
    
    # Get attention weights for visualization
    clean_output = run_inference(model, tokenizer, wild_type, device)
    attentions = clean_output['attentions']
    
    # Plot attention heatmap for the critical layer
    print(f"\nPlotting attention weights for Layer {max_effect_layer}...")
    # Average across heads to get a single heatmap
    layer_attention = attentions[max_effect_layer]
    plot_attention_heatmap(layer_attention, max_effect_layer, head_idx=None, file_prefix=file_prefix)
    
    # Plot indirect effects across all layers
    print("Plotting indirect effects across all layers...")
    plot_indirect_effects(layer_effects, highlight_layer=max_effect_layer, file_prefix=file_prefix)
    
    # Plot logit comparisons
    print("Plotting logit comparisons...")
    compare_logits_by_position(
        detailed_result['original_logits'],
        detailed_result['patched_logits'],
        detailed_result['corrupt_logits'],
        file_prefix=file_prefix,
    )
    
    # ========================================================================
    # Step 4: Attention Head Circuit Analysis (Optional)
    # ========================================================================
    head_effects = None
    if run_head_analysis:
        print("\n" + "=" * 70)
        print(f"STEP 4: Circuit Analysis - Attention Heads in Layer {max_effect_layer}")
        print("=" * 70)
        
        head_effects = analyze_heads_in_layer(
            model,
            tokenizer,
            wild_type,
            mutant,
            all_corrupt_states,
            max_effect_layer,
            mutation_pos,
            device,
        )
        
        # Visualize head effects
        print("\nGenerating attention head visualization...")
        plot_head_effects(
            head_effects,
            max_effect_layer,
            file_prefix=file_prefix,
        )
    
    # ========================================================================
    # Step 5: Summary Statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nOriginal Logit Difference: {detailed_result['logit_difference_original']:.4f}")
    print(f"Patched Logit Difference: {detailed_result['logit_difference_patched']:.4f}")
    print(f"Indirect Effect: {detailed_result['indirect_effect']:.4f}")
    
    print(f"\nTop 5 Layers by Indirect Effect:")
    sorted_layers = sorted(layer_effects.items(), key=lambda x: x[1], reverse=True)
    for rank, (layer_idx, effect) in enumerate(sorted_layers[:5], 1):
        print(f"  {rank}. Layer {layer_idx}: {effect:.4f}")
    
    if head_effects:
        print(f"\nTop 5 Attention Heads in Layer {max_effect_layer}:")
        sorted_heads = sorted(head_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        for rank, (head_idx, effect) in enumerate(sorted_heads[:5], 1):
            print(f"  {rank}. Head {head_idx}: {effect:.4f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return model, tokenizer, device, layer_effects, head_effects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mechanistic Interpretability Analysis for Nucleotide Transformer v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Analyze malignant TP53 mutation (default)
  python main.py --type malignant
  
  # Analyze benign TP53 mutation
  python main.py --type benign
  
  # Use synthetic sequences instead of NCBI data
  python main.py --synthetic
  
  # Combine options
  python main.py --type benign --synthetic"""
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['malignant', 'benign'],
        default='malignant',
        help='Type of TP53 mutation to analyze (default: malignant)'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic sequences instead of fetching from NCBI'
    )
    
    parser.add_argument(
        '--skip-head-analysis',
        action='store_true',
        help='Skip the attention head circuit analysis (faster but less detailed)'
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    model, tokenizer, device, layer_effects, head_effects = main(
        use_ncbi=not args.synthetic,
        use_tp53=True,
        mutation_type=args.type,
        run_head_analysis=not args.skip_head_analysis
    )
