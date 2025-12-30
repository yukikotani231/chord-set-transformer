# CLAUDE.md

This file provides guidance for AI assistants working with this codebase.

## Project Overview

This project implements chord recognition using Set Transformer, treating musical chords as unordered sets of notes. The key insight is that a chord like C major ({C, E, G}) is the same regardless of note ordering, making Set Transformer's permutation-invariant architecture a natural fit.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format src tests

# Lint code
uv run ruff check src tests

# Type check
uv run mypy src

# Run all checks
uv run ruff format src tests && uv run ruff check src tests && uv run mypy src && uv run pytest
```

## Architecture

### Data Pipeline
- `src/data/chords.py` - Chord definitions (intervals, note mappings)
- `src/data/dataset.py` - Synthetic chord dataset for experiments
- `src/data/harte.py` - Harte notation parser (e.g., "C:maj7/5")
- `src/data/choco_dataset.py` - ChoCo real-world dataset loader

### Models
- `src/models/modules.py` - Set Transformer building blocks (MAB, SAB, ISAB, PMA)
- `src/models/classifier.py` - Chord classifier (Set Transformer and DeepSets baseline)

## Key Concepts

### Set Transformer Components
- **MAB (Multihead Attention Block)**: Basic attention + FFN block
- **SAB (Set Attention Block)**: Self-attention over set elements, O(n²)
- **ISAB (Induced Set Attention Block)**: Uses inducing points for O(nm) complexity
- **PMA (Pooling by Multihead Attention)**: Aggregates set to fixed-size output

### Chord Representation
- Input: Set of MIDI notes → feature vectors (pitch class one-hot + octave + relative position)
- Output: Chord class (e.g., maj, min, dim, aug, dom7, maj7, min7)

## Testing

Tests are in `tests/` directory. Run with:
```bash
uv run pytest -v
```

## Data

### Synthetic Dataset
- Generated programmatically
- 48 classes (12 roots × 4 triad types) or 8 simplified categories
- Useful for controlled experiments

### ChoCo Dataset
- Download: `cd data && git clone --depth 1 https://github.com/smashub/choco.git`
- 943k+ chord annotations from 18 sources
- 8 chord categories with imbalanced distribution

## Common Tasks

### Add a new chord type
1. Add intervals to `CHORD_TYPES` in `src/data/chords.py`
2. Update `HARTE_QUALITY_MAP` in `src/data/harte.py` if needed

### Modify model architecture
- Edit `ChordSetTransformer` in `src/models/classifier.py`
- Key parameters: `d_model`, `num_heads`, `num_layers`, `use_isab`

### Train on synthetic data
```python
from src.data.dataset import EnhancedChordDataset, collate_fn
from src.models.classifier import ChordSetTransformer
dataset = EnhancedChordDataset(num_samples=10000)
model = ChordSetTransformer(input_dim=14, num_classes=48)
```
