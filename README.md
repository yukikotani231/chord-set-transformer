# Chord Set Transformer

Chord recognition using Set Transformer - treating chords as unordered sets of notes.

## Concept

A chord is fundamentally a **set** of notes where order doesn't matter:
- C major = {C, E, G} = {E, G, C} = {G, C, E}
- In MIDI: {60, 64, 67} in any order

This makes [Set Transformer](https://arxiv.org/abs/1810.00825) (Lee et al., 2019) a natural fit for chord classification, as it's designed specifically for permutation-invariant set processing.

## Features

- **Set Transformer architecture** for permutation-invariant chord classification
- **Synthetic chord dataset** for controlled experiments
- **ChoCo dataset integration** for real-world chord data (943k+ chord annotations)
- **DeepSets baseline** for comparison

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chord-set-transformer.git
cd chord-set-transformer

# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --extra dev
```

## Quick Start

### Synthetic Data Experiment

```python
from src.data.dataset import EnhancedChordDataset, collate_fn
from src.models.classifier import ChordSetTransformer
from torch.utils.data import DataLoader

# Create dataset (48 classes: 12 roots x 4 chord types)
dataset = EnhancedChordDataset(num_samples=10000)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

# Create model
model = ChordSetTransformer(
    input_dim=dataset.feature_dim,
    num_classes=dataset.num_classes,
    d_model=64,
    num_heads=4,
    num_layers=2,
)

# Train...
```

### ChoCo Real Data

```bash
# Download ChoCo dataset
cd data && git clone --depth 1 https://github.com/smashub/choco.git
```

```python
from src.data.choco_dataset import ChoCoDataset, collate_fn

# Load ChoCo dataset (8 chord categories)
dataset = ChoCoDataset(data_dir="data/choco", split="train")
```

## Results

### Synthetic Data (48 classes)

| Model | Test Accuracy |
|-------|---------------|
| Set Transformer | **93.5%** |
| DeepSets | 91.7% |

## Project Structure

```
chord-set-transformer/
├── src/
│   ├── data/
│   │   ├── chords.py         # Chord definitions
│   │   ├── dataset.py        # Synthetic dataset
│   │   ├── choco_dataset.py  # ChoCo dataset loader
│   │   ├── harte.py          # Harte notation parser
│   │   └── download.py       # Dataset download utilities
│   └── models/
│       ├── modules.py        # SAB, ISAB, PMA modules
│       └── classifier.py     # Chord classifier models
├── tests/
└── pyproject.toml
```

## References

- [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825) (Lee et al., ICML 2019)
- [ChoCo: the Chord Corpus](https://github.com/smashub/choco) (de Berardinis et al., Scientific Data 2023)

## License

MIT License
