"""ChoCo dataset loader for chord classification."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .harte import ParsedChord, parse_harte

# Simplified chord categories for classification
CHORD_CATEGORIES = [
    "maj",
    "min",
    "dim",
    "aug",
    "dom7",
    "maj7",
    "min7",
    "other",
]

CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CHORD_CATEGORIES)}


class ChoCoDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset from ChoCo chord corpus.

    Loads chord annotations from JAMS files and generates
    MIDI note sets for each chord.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        use_inversions: bool = True,
        octave_range: tuple[int, int] = (3, 6),
        max_samples: int | None = None,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to ChoCo dataset (contains partitions/).
            split: 'train', 'val', or 'test'.
            use_inversions: Whether to apply random inversions.
            octave_range: Range of octaves for note generation.
            max_samples: Maximum samples to load (for debugging).
            seed: Random seed.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_inversions = use_inversions
        self.octave_range = octave_range
        self.rng = np.random.default_rng(seed)

        # Load chord annotations
        self.chords = self._load_chords(max_samples)
        print(f"Loaded {len(self.chords)} chords for {split}")

        # Split data
        self._apply_split()

        self._feature_dim = 14  # Same as EnhancedChordDataset

    def _load_chords(self, max_samples: int | None) -> list[ParsedChord]:
        """Load all chord annotations from JAMS files."""
        chords: list[ParsedChord] = []
        partitions_dir = self.data_dir / "partitions"

        if not partitions_dir.exists():
            raise FileNotFoundError(f"Partitions directory not found: {partitions_dir}")

        # Find all JAMS files
        jams_files = list(partitions_dir.glob("*/choco/jams/*.jams"))
        print(f"Found {len(jams_files)} JAMS files")

        for jams_path in jams_files:
            if max_samples and len(chords) >= max_samples:
                break

            try:
                with open(jams_path) as f:
                    data = json.load(f)

                # Look for chord annotations (prefer Harte, fall back to Roman)
                for ann in data.get("annotations", []):
                    namespace = ann.get("namespace", "")
                    if namespace not in ["chord_harte", "chord", "chord_roman"]:
                        continue

                    for obs in ann.get("data", []):
                        value = obs.get("value", "")
                        if not value or value in ["N", "X", ""]:
                            continue

                        parsed = parse_harte(value)
                        if parsed and parsed.intervals:
                            chords.append(parsed)

                            if max_samples and len(chords) >= max_samples:
                                break

            except (json.JSONDecodeError, KeyError):
                continue

        return chords

    def _apply_split(self) -> None:
        """Apply train/val/test split."""
        n = len(self.chords)
        indices = self.rng.permutation(n)

        # 80/10/10 split
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if self.split == "train":
            self.indices = indices[:train_end]
        elif self.split == "val":
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]

        print(f"Split '{self.split}': {len(self.indices)} samples")

    def _note_to_features(self, note: int, bass_note: int) -> np.ndarray:
        """Convert MIDI note to feature vector."""
        features = np.zeros(self._feature_dim, dtype=np.float32)

        # Pitch class one-hot (dims 0-11)
        pitch_class = note % 12
        features[pitch_class] = 1.0

        # Octave normalized (dim 12)
        octave = note // 12
        features[12] = octave / 10.0

        # Relative position to bass (dim 13)
        features[13] = (note - bass_note) / 24.0

        return features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chord = self.chords[self.indices[idx]]

        # Random octave
        octave = self.rng.integers(self.octave_range[0], self.octave_range[1])

        # Get MIDI notes
        notes = chord.to_midi_notes(octave)

        # Apply random inversion
        if self.use_inversions and self.rng.random() > 0.5:
            num_inversions = self.rng.integers(1, len(notes))
            notes = list(notes)
            for _ in range(num_inversions):
                i = self.rng.integers(0, len(notes))
                if self.rng.random() > 0.5:
                    notes[i] += 12
                else:
                    notes[i] -= 12
            notes = sorted(notes)

        # Create features
        bass_note = min(notes)
        features = np.array([self._note_to_features(n, bass_note) for n in notes], dtype=np.float32)

        # Get category label
        category = chord.simplified_quality
        if category not in CATEGORY_TO_IDX:
            category = "other"
        label = CATEGORY_TO_IDX[category]

        return {
            "features": torch.from_numpy(features),
            "label": torch.tensor(label, dtype=torch.long),
            "num_notes": torch.tensor(len(notes)),
            "root": torch.tensor(chord.root),
        }

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_classes(self) -> int:
        return len(CHORD_CATEGORIES)

    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of chord categories."""
        counter: Counter[str] = Counter()
        for idx in self.indices:
            chord = self.chords[idx]
            category = chord.simplified_quality
            if category not in CATEGORY_TO_IDX:
                category = "other"
            counter[category] += 1
        return dict(counter)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for variable-length chord sets."""
    max_notes = int(max(sample["num_notes"].item() for sample in batch))
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[-1]

    features = torch.zeros(batch_size, max_notes, feature_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.zeros(batch_size, max_notes, dtype=torch.bool)
    roots = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        n = int(sample["num_notes"].item())
        features[i, :n] = sample["features"]
        labels[i] = sample["label"]
        mask[i, :n] = True
        roots[i] = sample["root"]

    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "roots": roots,
        "num_notes": torch.tensor([s["num_notes"].item() for s in batch]),
    }
