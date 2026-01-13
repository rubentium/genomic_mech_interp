# Genomic Mechanistic Interpretability

A comprehensive mechanistic interpretability pipeline for analyzing how the Nucleotide Transformer v2 (500M parameter) model processes DNA mutations. This project implements activation patching, attention analysis, and circuit discovery techniques to understand the internal mechanisms by which genomic foundation models detect and respond to genetic variants.

## Overview

This toolkit enables researchers to:

- **Analyze Real Mutations**: Fetch and analyze real genomic sequences from NCBI, including cancer-associated and benign TP53 variants
- **Perform Activation Patching**: Identify which model layers are responsible for detecting specific mutations using causal intervention techniques
- **Circuit Discovery**: Decompose model behavior down to individual attention heads to understand mutation detection circuits
- **Visualize Model Internals**: Generate attention heatmaps, indirect effect plots, and logit comparisons

## Features

### 🧬 Real Genomic Data Integration
- Direct integration with NCBI for fetching real genomic sequences
- Pre-configured analysis of TP53 mutations (both pathogenic and benign)
- Automatic sequence alignment and mutation verification

### 🔬 Mechanistic Analysis
- **Layer-wise Activation Patching**: Identifies which transformer layers contribute to mutation detection
- **Attention Head Analysis**: Pinpoints individual attention heads responsible for detecting variants
- **Logit Attribution**: Quantifies model confidence in wild-type vs mutant predictions

### 📊 Comprehensive Visualization
- Attention weight heatmaps with mutation highlighting
- Indirect effect bar charts across all layers
- Head-level circuit analysis plots
- Position-wise logit comparisons

## Project Structure

```
genomic_mech_interp/
├── main.py          # Main execution script with CLI
├── patching.py      # Core mechanistic interpretability functions
├── plotter.py       # Visualization and plotting utilities
├── requirements.txt # Python dependencies
└── README.md        # This file
```

### Module Descriptions

- **`main.py`**: Orchestrates the complete analysis pipeline, handles command-line arguments, and coordinates data fetching, model inference, and visualization generation.

- **`patching.py`**: Contains the core mechanistic interpretability logic including:
  - Model and tokenizer loading
  - NCBI sequence fetching utilities
  - Activation hook system for capturing layer outputs
  - Single-head and full-layer activation patching
  - Logit difference calculations

- **`plotter.py`**: Provides visualization functions for:
  - Attention weight heatmaps
  - Layer-wise indirect effect bar charts
  - Attention head contribution plots
  - Logit comparison visualizations

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, CPU will work but is slower)

### Setup

1. Clone the repository:
```bash
cd genomic_mech_interp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure NCBI email in `patching.py`:
```python
# In patching.py, update:
Entrez.email = "your_email@example.com"
```

## Usage

### Basic Usage

Analyze a malignant TP53 mutation (default):
```bash
python main.py
```

### Command-Line Options

Analyze a benign TP53 mutation:
```bash
python main.py --type benign
```

Use synthetic test sequences (no NCBI connection required):
```bash
python main.py --synthetic
```

Skip detailed head analysis for faster execution:
```bash
python main.py --skip-head-analysis
```

Combine options:
```bash
python main.py --type benign --skip-head-analysis
```

### Output Files

The analysis generates several visualization files:

- `{type}_attention_layer_{N}.png` - Attention heatmap for critical layer
- `{type}_indirect_effects_summary.png` - Layer-wise indirect effects
- `{type}_head_effects_layer_{N}.png` - Head-level circuit analysis
- `{type}_logits_comparison.png` - Position-wise logit comparisons

Where `{type}` is either `malignant_` or `benign_` for NCBI data, or empty for synthetic sequences.

## Example Analysis Workflow

```python
# Import modules
from patching import ModelLoader, NCBISequenceFetcher, analyze_all_layers
from plotter import plot_indirect_effects

# Load model
model, tokenizer, device = ModelLoader.load_model()

# Fetch real genomic data
wild_type, mutant, mutation_pos = NCBISequenceFetcher.fetch_tp53_mutation()

# Analyze all layers
layer_effects = analyze_all_layers(
    model, tokenizer, wild_type, mutant, mutation_pos, device
)

# Visualize results
plot_indirect_effects(layer_effects)
```

## Technical Details

### Model Architecture
- **Model**: Nucleotide Transformer v2 (500M parameters)
- **Architecture**: BERT-style transformer with ESM backbone
- **Layers**: 33 transformer layers
- **Attention Heads**: 20 heads per layer
- **Context Window**: 1000 tokens (≈6000 base pairs)

### Activation Patching Methodology

1. **Baseline Capture**: Run clean (wild-type) sequence through model
2. **Corrupt Capture**: Run mutant sequence and store all layer activations
3. **Intervention**: For each layer, replace clean activations with mutant activations
4. **Effect Measurement**: Calculate change in logit difference (indirect effect)
5. **Attribution**: Layers with large indirect effects are critical for mutation detection

### Metrics

- **Logit Difference**: `logit(WT) - logit(MUT)` at mutation position
- **Indirect Effect**: Change in logit difference after patching
- **Head Attribution**: Per-head contribution to overall layer effect

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genomic_mech_interp,
  title = {Mechanistic Interpretability for Nucleotide Transformer Models},
  author = {Ruben Navasardyan},
  year = {Jan. 2026},
  url = {https://github.com/yourusername/genomic-mech-interp}
}
```

## References

- **Nucleotide Transformer**: Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics" (2023)
- **Activation Patching**: Meng et al., "Locating and Editing Factual Associations in GPT" (2022)
- **Mechanistic Interpretability**: Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

- InstaDeep AI for the Nucleotide Transformer model
- Anthropic and OpenAI for mechanistic interpretability research
- The Biopython team for NCBI integration tools
