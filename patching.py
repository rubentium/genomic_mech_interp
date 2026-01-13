"""
Activation Patching and Model Inference Module

This module contains all the core mechanistic interpretability functionality:
- Model and sequence loading
- Activation capture via hooks
- Logit difference calculation
- Layer and head-level activation patching
"""

import os
import warnings
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from Bio import Entrez, SeqIO

warnings.filterwarnings("ignore")


# ============================================================================
# Model Setup and Loading
# ============================================================================

class ModelLoader:
    """Utility class for loading the Nucleotide Transformer v2 (500M) model."""
    
    MODEL_NAME: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    
    @staticmethod
    def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
        """
        Load the Nucleotide Transformer v2 model, tokenizer, and move to GPU if available.
        
        Returns:
            Tuple containing:
                - model: The pre-trained model with output_hidden_states and output_attentions enabled
                - tokenizer: The corresponding tokenizer
                - device: The device (cuda or cpu) where the model is loaded
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {device}")
        print(f"Using HF cache directory: {os.environ.get('HF_HOME', 'default')}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ModelLoader.MODEL_NAME)
        
        # Load model with hidden states and attention outputs enabled
        model = AutoModelForMaskedLM.from_pretrained(
            ModelLoader.MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
            trust_remote_code=True,  # Required for nucleotide-transformer custom code
        )
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully with {model.config.num_hidden_layers} layers")
        return model, tokenizer, device


class NCBISequenceFetcher:
    """Utility class for fetching genomic sequences from NCBI."""
    
    # Configure NCBI Entrez email (required by NCBI)
    Entrez.email = "your_email@example.com"
    
    @staticmethod
    def fetch_sequence_around_mutation(
        chrom_accession: str,
        mutation_pos: int,
        wt_nucleotide: str,
        mut_nucleotide: str,
        flanking_bp: int = 500,
    ) -> Tuple[str, str, int]:
        """
        Fetch wild-type and mutant sequences from NCBI around a specific mutation.
        
        Args:
            chrom_accession: NCBI chromosome accession (e.g., "NC_000017.11" for chr17 hg38)
            mutation_pos: 1-based genomic coordinate of the mutation
            wt_nucleotide: Expected wild-type nucleotide at mutation site
            mut_nucleotide: Mutant nucleotide to substitute
            flanking_bp: Number of base pairs to fetch on each side of mutation
            
        Returns:
            Tuple of (wild_type_sequence, mutant_sequence, mutation_position_in_sequence)
        """
        start = mutation_pos - flanking_bp
        end = mutation_pos + flanking_bp
        
        print(f"\nFetching sequence from NCBI...")
        print(f"Chromosome: {chrom_accession}")
        print(f"Region: {start:,} to {end:,} ({end-start+1} bp)")
        print(f"Mutation: {wt_nucleotide}->{mut_nucleotide} at position {mutation_pos:,}")
        
        try:
            handle = Entrez.efetch(
                db="nucleotide",
                id=chrom_accession,
                rettype="fasta",
                seq_start=start,
                seq_stop=end
            )
            record = SeqIO.read(handle, "fasta")
            handle.close()
            
            wild_type = str(record.seq).upper()
            
            # The mutation is at the center of our window
            mutation_index = flanking_bp
            
            # Verify the wild-type nucleotide matches expectation
            actual_wt = wild_type[mutation_index]
            print(f"\nVerification:")
            print(f"  Expected WT nucleotide: {wt_nucleotide}")
            print(f"  Actual WT nucleotide:   {actual_wt}")
            
            if actual_wt != wt_nucleotide:
                print(f"  WARNING: Mismatch! Using actual nucleotide '{actual_wt}' as wild-type.")
                wt_nucleotide = actual_wt
            
            # Create mutant sequence
            mutant_list = list(wild_type)
            mutant_list[mutation_index] = mut_nucleotide
            mutant = "".join(mutant_list)
            
            print(f"\nSequence context around mutation (position {mutation_index}):")
            print(f"  WT:  ...{wild_type[mutation_index-10:mutation_index+11]}...")
            print(f"  MUT: ...{mutant[mutation_index-10:mutation_index+11]}...")
            print(f"           {' '*10}^")
            
            return wild_type, mutant, mutation_index
            
        except Exception as e:
            print(f"Error fetching sequence from NCBI: {e}")
            raise
    
    @staticmethod
    def fetch_tp53_mutation(flanking_bp: int = 500) -> Tuple[str, str, int]:
        """
        Fetch the TP53 G->T mutation at chr17:7673803 (hg38).
        
        This is a well-studied cancer-associated mutation in the TP53 tumor suppressor gene.
        
        Args:
            flanking_bp: Number of base pairs to fetch on each side
            
        Returns:
            Tuple of (wild_type_sequence, mutant_sequence, mutation_position_in_sequence)
        """
        return NCBISequenceFetcher.fetch_sequence_around_mutation(
            chrom_accession="NC_000017.11",  # Chromosome 17 (hg38)
            mutation_pos=7673803,
            wt_nucleotide="G",
            mut_nucleotide="T",
            flanking_bp=flanking_bp
        )

    @staticmethod
    def fetch_tp53_benign_control(flanking_bp: int = 500) -> Tuple[str, str, int]:
        """
        Fetch the TP53 C->G benign polymorphism (p.P72R) at chr17:7676154 (hg38).
        
        This serves as a benign control to contrast against pathogenic mutations.
        """
        return NCBISequenceFetcher.fetch_sequence_around_mutation(
            chrom_accession="NC_000017.11", 
            mutation_pos=7676154,
            wt_nucleotide="G",
            mut_nucleotide="C",
            flanking_bp=flanking_bp
        )

# ============================================================================
# Inference Function with Output Capture
# ============================================================================

def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequence: str,
    device: torch.device,
) -> Dict:
    """
    Run inference on a DNA sequence and capture model outputs.
    
    Args:
        model: The pre-trained model
        tokenizer: The tokenizer for the model
        sequence: DNA sequence string (ACGT)
        device: Device to run inference on
        
    Returns:
        Dictionary containing:
            - 'logits': Model logits [1, seq_len, vocab_size]
            - 'hidden_states': Tuple of hidden states from all layers
            - 'attentions': Tuple of attention weights from all layers
            - 'input_ids': Tokenized input
    """
    # Tokenize the sequence
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=False,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return {
        'logits': outputs.logits,
        'hidden_states': outputs.hidden_states,
        'attentions': outputs.attentions,
        'input_ids': inputs['input_ids'],
    }


# ============================================================================
# Logit Difference Analysis
# ============================================================================

def calculate_logit_difference(
    logits: torch.Tensor,
    token_index: int,
    wt_token_id: int,
    mut_token_id: int,
    device: torch.device,
) -> float:
    """
    Calculate the genomic logit difference at a specific mutation site.
    
    This function computes: Logit(WildType_Token) - Logit(Mutant_Token) 
    at the specific token_index, providing a precise measure of the model's
    preference between wild-type and mutant nucleotides.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        token_index: The token position corresponding to the mutation site
        wt_token_id: Token ID for the wild-type nucleotide
        mut_token_id: Token ID for the mutant nucleotide
        device: Device to ensure tensors are on correct device
        
    Returns:
        Scalar logit difference value (WT_logit - MUT_logit)
    """
    # Ensure logits are on the correct device
    logits = logits.to(device)
    
    # Extract logits at the specific token position
    # Shape: [batch_size, vocab_size] -> [vocab_size]
    position_logits = logits[0, token_index, :]
    
    # Get logits for wild-type and mutant tokens
    wt_logit = position_logits[wt_token_id].item()
    mut_logit = position_logits[mut_token_id].item()
    
    # Calculate genomic logit difference: WT - MUT
    # Positive values indicate preference for wild-type
    logit_diff = wt_logit - mut_logit
    return logit_diff


# ============================================================================
# Activation Patching - Core Mechanistic Interpretability Task
# ============================================================================

def patch_single_head(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    clean_sequence: str,
    corrupt_sequence: str,
    all_corrupt_states: tuple,
    target_layer: int,
    target_head: int,
    mutation_pos: int,
    device: torch.device,
) -> Dict:
    """
    Patch a single attention head within a specific layer.
    
    This function isolates the contribution of individual attention heads by:
    1. Running the clean sequence through the model
    2. Replacing only the activations from a specific head with corrupt activations
    3. Measuring how much this single-head intervention affects the output
    
    Args:
        model: The pre-trained model
        tokenizer: Tokenizer for the model
        clean_sequence: Wild-type DNA sequence
        corrupt_sequence: Mutant DNA sequence
        all_corrupt_states: Pre-captured hidden states from corrupt sequence
        target_layer: Which layer contains the head to patch
        target_head: Which specific head to patch (0 to num_heads-1)
        mutation_pos: Character position of the mutation
        device: Device to run on
        
    Returns:
        Dictionary containing logits and indirect effect
    """
    # Get model configuration
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    # Tokenize and map mutation position to token index
    clean_encoding = tokenizer(clean_sequence, return_tensors="pt", truncation=False).to(device)
    input_ids = clean_encoding.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Find token index for mutation
    token_index = None
    char_count = 0
    for idx, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens:
            continue
        token_clean = token.replace('Ġ', '').replace('##', '').replace('▁', '')
        if char_count <= mutation_pos < char_count + len(token_clean):
            token_index = idx
            break
        char_count += len(token_clean)
    
    # Get token IDs for logit calculation
    wt_token_id = tokenizer.convert_tokens_to_ids(clean_sequence[mutation_pos])
    mut_token_id = tokenizer.convert_tokens_to_ids(corrupt_sequence[mutation_pos])
    if isinstance(wt_token_id, list):
        wt_token_id = wt_token_id[0]
    if isinstance(mut_token_id, list):
        mut_token_id = mut_token_id[0]
    
    # Baseline: Clean sequence
    clean_output = run_inference(model, tokenizer, clean_sequence, device)
    clean_logits = clean_output['logits'].detach()
    original_logit_diff = calculate_logit_difference(
        clean_logits, token_index, wt_token_id, mut_token_id, device
    )
    
    # Extract corrupt hidden states for the target layer
    corrupt_hidden_state = all_corrupt_states[target_layer].clone()
    
    def head_patch_hook(module, input, output):
        """
        Hook that patches only a specific attention head.
        
        ESM layers output (hidden_states, attentions). We need to:
        1. Take the clean hidden states
        2. Reshape to expose individual heads
        3. Replace only target_head with corrupt activations
        4. Reshape back and return
        """
        if isinstance(output, tuple):
            clean_hidden = output[0]
        else:
            clean_hidden = output
        
        batch_size, seq_len, hidden_size = clean_hidden.shape
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        clean_reshaped = clean_hidden.view(batch_size, seq_len, num_heads, head_dim)
        corrupt_reshaped = corrupt_hidden_state.view(batch_size, seq_len, num_heads, head_dim)
        
        # Replace only the target head
        clean_reshaped[:, :, target_head, :] = corrupt_reshaped[:, :, target_head, :]
        
        # Reshape back to [batch, seq_len, hidden_size]
        patched_hidden = clean_reshaped.view(batch_size, seq_len, hidden_size)
        
        if isinstance(output, tuple):
            return (patched_hidden,) + output[1:]
        return (patched_hidden,)
    
    # Register hook on target layer
    target_module = model.esm.encoder.layer[target_layer]
    handle = target_module.register_forward_hook(head_patch_hook)
    
    try:
        with torch.no_grad():
            patched_outputs = model(**clean_encoding)
            patched_logits = patched_outputs.logits.detach()
    finally:
        handle.remove()
    
    # Calculate effect of this single head
    patched_logit_diff = calculate_logit_difference(
        patched_logits, token_index, wt_token_id, mut_token_id, device
    )
    indirect_effect = original_logit_diff - patched_logit_diff
    
    return {
        'original_logits': clean_logits,
        'patched_logits': patched_logits,
        'logit_difference_original': original_logit_diff,
        'logit_difference_patched': patched_logit_diff,
        'indirect_effect': indirect_effect,
        'token_index': token_index,
        'head_index': target_head,
    }


def analyze_heads_in_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    clean_sequence: str,
    corrupt_sequence: str,
    all_corrupt_states: tuple,
    target_layer: int,
    mutation_pos: int,
    device: torch.device,
) -> Dict[int, float]:
    """
    Analyze all attention heads within a specific layer.
    
    This reveals which heads are most responsible for detecting the mutation.
    
    Args:
        model: The pre-trained model
        tokenizer: Tokenizer
        clean_sequence: Wild-type sequence
        corrupt_sequence: Mutant sequence
        all_corrupt_states: Pre-captured corrupt hidden states
        target_layer: Which layer to analyze
        mutation_pos: Character position of mutation
        device: Device to run on
        
    Returns:
        Dictionary mapping head_index -> indirect_effect
    """
    num_heads = model.config.num_attention_heads
    head_effects = {}
    
    print(f"\n{'='*70}")
    print(f"Attention Head Analysis - Layer {target_layer}")
    print(f"Analyzing {num_heads} attention heads...")
    print(f"{'='*70}")
    
    for head_idx in range(num_heads):
        print(f"\nPatching Head {head_idx}/{num_heads-1}...", end=" ")
        
        result = patch_single_head(
            model,
            tokenizer,
            clean_sequence,
            corrupt_sequence,
            all_corrupt_states,
            target_layer,
            head_idx,
            mutation_pos,
            device,
        )
        
        head_effects[head_idx] = result['indirect_effect']
        print(f"Indirect Effect: {result['indirect_effect']:.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Top 5 Most Critical Heads:")
    sorted_heads = sorted(head_effects.items(), key=lambda x: abs(x[1]), reverse=True)
    for rank, (head_idx, effect) in enumerate(sorted_heads[:5], 1):
        print(f"  {rank}. Head {head_idx}: {effect:.4f}")
    print(f"{'='*70}")
    
    return head_effects


def patch_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    clean_sequence: str,
    corrupt_sequence: str,
    all_corrupt_states: tuple,
    target_layer: int,
    mutation_pos: int,
    device: torch.device,
) -> Dict:
    """
    Fixed Activation Patching for Nucleotide Transformer (ESM architecture).
    """
    print(f"\n{'='*70}")
    print(f"Genomic Activation Patching Analysis - Layer {target_layer}")
    print(f"{'='*70}")
    
    clean_encoding = tokenizer(clean_sequence, return_tensors="pt", truncation=False).to(device)
    input_ids = clean_encoding.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Character to Token Mapping
    token_index = None
    char_count = 0
    for idx, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens: continue
        token_clean = token.replace('Ġ', '').replace('##', '').replace('▁', '')
        if char_count <= mutation_pos < char_count + len(token_clean):
            token_index = idx
            break
        char_count += len(token_clean)

    # Get Token IDs for Logit Math
    wt_token_id = tokenizer.convert_tokens_to_ids(clean_sequence[mutation_pos])
    mut_token_id = tokenizer.convert_tokens_to_ids(corrupt_sequence[mutation_pos])
    if isinstance(wt_token_id, list): wt_token_id = wt_token_id[0]
    if isinstance(mut_token_id, list): mut_token_id = mut_token_id[0]

    # Baseline Inference
    clean_output = run_inference(model, tokenizer, clean_sequence, device)
    clean_logits = clean_output['logits'].detach()
    original_logit_diff = calculate_logit_difference(clean_logits, token_index, wt_token_id, mut_token_id, device)

    print(f"Applying patch at Layer {target_layer}...")

    def patch_hook_fn(module, input, output):
        # ESM layers expect a tuple return: (hidden_states, attentions)
        # We must provide our patched state at index 0 of a tuple
        patched_hidden_states = all_corrupt_states[target_layer]
        
        if isinstance(output, tuple):
            # If the model expects multiple outputs (like attentions), preserve them
            return (patched_hidden_states,) + output[1:]
        return (patched_hidden_states,)

    # Register hook on the specific ESM layer
    target_module = model.esm.encoder.layer[target_layer]
    handle = target_module.register_forward_hook(patch_hook_fn)

    try:
        with torch.no_grad():
            # Run clean sequence with the mutation-site 'corrupt' patch active
            patched_outputs = model(**clean_encoding)
            patched_logits = patched_outputs.logits.detach()
    finally:
        # ALWAYS remove the hook, even if inference crashes
        handle.remove()

    patched_logit_diff = calculate_logit_difference(patched_logits, token_index, wt_token_id, mut_token_id, device)
    indirect_effect = original_logit_diff - patched_logit_diff

    print(f"Original Logit Diff: {original_logit_diff:.4f}")
    print(f"Patched Logit Diff:  {patched_logit_diff:.4f}")
    print(f"Indirect Effect:     {indirect_effect:.4f}")
    
    corrupt_baseline = run_inference(model, tokenizer, corrupt_sequence, device)
    corrupt_logits_tensor = corrupt_baseline['logits']

    return {
        'original_logits': clean_logits,
        'patched_logits': patched_logits,
        'corrupt_logits': corrupt_logits_tensor,
        'logit_difference_original': original_logit_diff,
        'logit_difference_patched': patched_logit_diff,
        'indirect_effect': indirect_effect,
        'token_index': token_index
    }


def analyze_all_layers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    clean_sequence: str,
    corrupt_sequence: str,
    mutation_pos: int,
    device: torch.device,
) -> Dict:
    """
    Fixed Orchestrator: Captures baseline once, then patches each layer.
    """
    num_layers = model.config.num_hidden_layers
    layer_effects = {}
    
    # --- Capture the mutant 'world' once ---
    print("Pre-capturing mutant activations for all layers...")
    corrupt_output = run_inference(model, tokenizer, corrupt_sequence, device)
    all_corrupt_states = corrupt_output['hidden_states'] 

    # --- Now pass those states into the patching function ---
    for layer_idx in range(num_layers):
        result = patch_activations(
            model, 
            tokenizer, 
            clean_sequence, 
            corrupt_sequence, 
            all_corrupt_states,
            layer_idx, 
            mutation_pos, 
            device
        )
        layer_effects[layer_idx] = result['indirect_effect']
        print(f"Layer {layer_idx}: Indirect Effect = {result['indirect_effect']:.4f}")
    
    return layer_effects
