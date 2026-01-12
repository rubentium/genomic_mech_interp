# Migration Guide: From mechanistic_interp.py to Modular Structure

## Quick Reference

### Old Usage
```bash
python mechanistic_interp.py --type benign
```

### New Usage
```bash
python main.py --type benign
```

## Import Changes

### Before (monolithic)
```python
# Everything was in one file
from mechanistic_interp import ModelLoader, plot_attention_heatmap
```

### After (modular)
```python
# Import from specific modules
from patching import ModelLoader, NCBISequenceFetcher, analyze_all_layers
from plotter import plot_attention_heatmap, plot_indirect_effects
```

## Module Organization

### patching.py - Core Analysis
Use this module for:
- Loading models and tokenizers
- Fetching genomic sequences from NCBI
- Running model inference
- Performing activation patching
- Analyzing layers and attention heads

```python
from patching import (
    ModelLoader,
    NCBISequenceFetcher,
    run_inference,
    analyze_all_layers,
    analyze_heads_in_layer,
)
```

### plotter.py - Visualization
Use this module for:
- Creating attention heatmaps
- Plotting indirect effects
- Visualizing head contributions
- Comparing logits across positions

```python
from plotter import (
    plot_attention_heatmap,
    plot_indirect_effects,
    plot_head_effects,
    compare_logits_by_position,
)
```

### main.py - Orchestration
Use this module for:
- Running complete analysis pipelines
- Command-line interface
- Example workflows

```bash
# Command-line usage
python main.py --type benign --skip-head-analysis

# Or import the main function
from main import main
model, tokenizer, device, layer_effects, head_effects = main(
    use_ncbi=True,
    mutation_type='benign',
    run_head_analysis=True
)
```

## Breaking Changes

None! All functionality has been preserved. The only changes are:
1. Different file to execute (`main.py` instead of `mechanistic_interp.py`)
2. Imports come from specific modules
3. Better organized and more maintainable code

## Benefits of New Structure

1. **Modularity**: Import only what you need
2. **Maintainability**: Each file has a clear purpose
3. **Testability**: Easier to write unit tests for specific functions
4. **Readability**: Smaller files are easier to navigate
5. **Scalability**: Easy to add new features to appropriate modules

## File Sizes Comparison

| Old Structure | Size | New Structure | Size |
|--------------|------|---------------|------|
| mechanistic_interp.py | 42KB | main.py | 11KB |
| | | patching.py | 24KB |
| | | plotter.py | 8.5KB |

Total code is the same, just better organized!

## Git Repository

The new structure is version controlled:
```bash
git log --oneline
git status
```

## Backward Compatibility

The original `mechanistic_interp.py` is still present in the directory if you need it for reference, but it's not tracked by git. You can safely delete it once you've verified the new structure works for your use case.

## Questions?

Check the README.md for comprehensive documentation, usage examples, and technical details.
