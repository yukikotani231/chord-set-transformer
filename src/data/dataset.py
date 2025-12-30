"""PyTorch Dataset for chord classification."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .chords import (
    TRIAD_TYPES,
    Chord,
    get_chord_class_index,
)


class SyntheticChordDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset of chords for classification.

    Generates chords with various augmentations:
    - Different octaves
    - Inversions (reordering notes)
    - Optional velocity variations
    - Optional note omissions/additions
    """

    def __init__(
        self,
        num_samples: int = 10000,
        chord_types: dict[str, list[int]] | None = None,
        use_inversions: bool = True,
        use_octave_variations: bool = True,
        octave_range: tuple[int, int] = (3, 6),
        add_noise: bool = False,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            num_samples: Number of samples to generate.
            chord_types: Chord types to use (default: TRIAD_TYPES).
            use_inversions: Whether to use chord inversions.
            use_octave_variations: Whether to vary octaves.
            octave_range: Range of octaves to use.
            add_noise: Whether to add random notes as noise.
            seed: Random seed.
        """
        self.num_samples = num_samples
        self.chord_types = chord_types or TRIAD_TYPES
        self.use_inversions = use_inversions
        self.use_octave_variations = use_octave_variations
        self.octave_range = octave_range
        self.add_noise = add_noise
        self.rng = np.random.default_rng(seed)

        self.num_classes = 12 * len(self.chord_types)

        # Pre-generate samples
        self._samples = self._generate_samples()

    def _generate_samples(self) -> list[dict[str, torch.Tensor]]:
        """Generate all samples."""
        samples = []

        for _ in range(self.num_samples):
            # Random root and chord type
            root = self.rng.integers(0, 12)
            chord_type = self.rng.choice(list(self.chord_types.keys()))

            # Get base octave
            if self.use_octave_variations:
                octave = int(self.rng.integers(self.octave_range[0], self.octave_range[1]))
            else:
                octave = 4

            # Create chord
            chord = Chord.from_root_and_type(int(root), chord_type, octave)
            notes = list(chord.notes)

            # Apply inversions (move some notes up/down an octave)
            if self.use_inversions and self.rng.random() > 0.5:
                num_inversions = self.rng.integers(1, len(notes))
                for _ in range(num_inversions):
                    idx = self.rng.integers(0, len(notes))
                    if self.rng.random() > 0.5:
                        notes[idx] += 12  # Up an octave
                    else:
                        notes[idx] -= 12  # Down an octave

            # Add noise notes (optional)
            if self.add_noise and self.rng.random() > 0.7:
                num_noise = int(self.rng.integers(1, 3))
                for _ in range(num_noise):
                    noise_note = int(self.rng.integers(36, 96))  # MIDI range
                    if noise_note not in notes:
                        notes.append(noise_note)

            # Create features: normalized MIDI notes
            # Normalize to [0, 1] range (MIDI 0-127)
            features = np.array(notes, dtype=np.float32) / 127.0

            # Get class label
            class_idx = get_chord_class_index(int(root), chord_type, self.chord_types)

            samples.append(
                {
                    "features": torch.from_numpy(features).unsqueeze(-1),  # (N, 1)
                    "label": torch.tensor(class_idx, dtype=torch.long),
                    "num_notes": torch.tensor(len(notes)),
                }
            )

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._samples[idx]

    @property
    def feature_dim(self) -> int:
        """Feature dimension per note."""
        return 1  # Just MIDI note number for now


class EnhancedChordDataset(Dataset[dict[str, torch.Tensor]]):
    """Enhanced chord dataset with richer features.

    Features per note:
    - Pitch class (one-hot, 12 dims)
    - Octave (normalized)
    - Relative position to bass note
    """

    def __init__(
        self,
        num_samples: int = 10000,
        chord_types: dict[str, list[int]] | None = None,
        use_inversions: bool = True,
        octave_range: tuple[int, int] = (3, 6),
        seed: int = 42,
    ):
        """Initialize dataset."""
        self.num_samples = num_samples
        self.chord_types = chord_types or TRIAD_TYPES
        self.use_inversions = use_inversions
        self.octave_range = octave_range
        self.rng = np.random.default_rng(seed)

        self.num_classes = 12 * len(self.chord_types)
        self._feature_dim = 14  # 12 (pitch class) + 1 (octave) + 1 (relative pos)

        self._samples = self._generate_samples()

    def _note_to_features(self, note: int, bass_note: int) -> np.ndarray:
        """Convert a MIDI note to feature vector.

        Args:
            note: MIDI note number
            bass_note: Bass note for relative position

        Returns:
            Feature vector of shape (14,)
        """
        features = np.zeros(self._feature_dim, dtype=np.float32)

        # Pitch class one-hot (dims 0-11)
        pitch_class = note % 12
        features[pitch_class] = 1.0

        # Octave normalized (dim 12)
        octave = note // 12
        features[12] = octave / 10.0  # Normalize roughly

        # Relative position to bass (dim 13)
        features[13] = (note - bass_note) / 24.0  # Normalize by 2 octaves

        return features

    def _generate_samples(self) -> list[dict[str, torch.Tensor]]:
        """Generate all samples."""
        samples = []

        for _ in range(self.num_samples):
            root = int(self.rng.integers(0, 12))
            chord_type = self.rng.choice(list(self.chord_types.keys()))
            octave = int(self.rng.integers(self.octave_range[0], self.octave_range[1]))

            chord = Chord.from_root_and_type(root, chord_type, octave)
            notes = list(chord.notes)

            # Apply inversions
            if self.use_inversions and self.rng.random() > 0.5:
                num_inversions = self.rng.integers(1, len(notes))
                for _ in range(num_inversions):
                    idx = self.rng.integers(0, len(notes))
                    if self.rng.random() > 0.5:
                        notes[idx] += 12
                    else:
                        notes[idx] -= 12

            # Sort and get bass
            notes_sorted = sorted(notes)
            bass_note = notes_sorted[0]

            # Create features
            features = np.array(
                [self._note_to_features(n, bass_note) for n in notes], dtype=np.float32
            )

            class_idx = get_chord_class_index(root, chord_type, self.chord_types)

            samples.append(
                {
                    "features": torch.from_numpy(features),
                    "label": torch.tensor(class_idx, dtype=torch.long),
                    "num_notes": torch.tensor(len(notes)),
                }
            )

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._samples[idx]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for variable-length chord sets.

    Args:
        batch: List of samples.

    Returns:
        Batched tensors with padding and mask.
    """
    max_notes = int(max(sample["num_notes"].item() for sample in batch))
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[-1]

    features = torch.zeros(batch_size, max_notes, feature_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.zeros(batch_size, max_notes, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = int(sample["num_notes"].item())
        features[i, :n] = sample["features"]
        labels[i] = sample["label"]
        mask[i, :n] = True

    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "num_notes": torch.tensor([s["num_notes"].item() for s in batch]),
    }
