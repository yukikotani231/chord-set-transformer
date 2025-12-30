"""Chord definitions and utilities."""

from dataclasses import dataclass

# MIDI note names (C4 = 60)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord type definitions as semitone intervals from root
CHORD_TYPES = {
    "maj": [0, 4, 7],  # Major triad
    "min": [0, 3, 7],  # Minor triad
    "dim": [0, 3, 6],  # Diminished triad
    "aug": [0, 4, 8],  # Augmented triad
    "maj7": [0, 4, 7, 11],  # Major 7th
    "min7": [0, 3, 7, 10],  # Minor 7th
    "dom7": [0, 4, 7, 10],  # Dominant 7th
    "dim7": [0, 3, 6, 9],  # Diminished 7th
    "hdim7": [0, 3, 6, 10],  # Half-diminished 7th
    "minmaj7": [0, 3, 7, 11],  # Minor-major 7th
    "aug7": [0, 4, 8, 10],  # Augmented 7th
    "sus2": [0, 2, 7],  # Suspended 2nd
    "sus4": [0, 5, 7],  # Suspended 4th
}

# Simplified chord types for initial experiments
TRIAD_TYPES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
}


@dataclass
class Chord:
    """Represents a chord."""

    root: int  # MIDI note number of root (0-11 for pitch class, or full MIDI)
    chord_type: str  # Key in CHORD_TYPES
    notes: tuple[int, ...]  # MIDI note numbers

    @property
    def name(self) -> str:
        """Get chord name like 'C:maj' or 'F#:min7'."""
        root_name = NOTE_NAMES[self.root % 12]
        return f"{root_name}:{self.chord_type}"

    @classmethod
    def from_root_and_type(
        cls,
        root: int,
        chord_type: str,
        octave: int = 4,
    ) -> "Chord":
        """Create a chord from root note and type.

        Args:
            root: Root pitch class (0-11, where 0=C)
            chord_type: Chord type key (e.g., 'maj', 'min')
            octave: Octave number (default 4, so C4=60)

        Returns:
            Chord instance.
        """
        if chord_type not in CHORD_TYPES:
            raise ValueError(f"Unknown chord type: {chord_type}")

        intervals = CHORD_TYPES[chord_type]
        base_note = 12 * (octave + 1) + root  # MIDI note number
        notes = tuple(base_note + interval for interval in intervals)

        return cls(root=root, chord_type=chord_type, notes=notes)


def get_chord_class_index(root: int, chord_type: str, chord_types: dict[str, list[int]]) -> int:
    """Get class index for a chord.

    Args:
        root: Root pitch class (0-11)
        chord_type: Chord type key
        chord_types: Dictionary of chord types to use

    Returns:
        Class index (0 to num_classes-1)
    """
    type_list = list(chord_types.keys())
    type_idx = type_list.index(chord_type)
    return root * len(chord_types) + type_idx


def get_chord_from_class_index(
    class_idx: int,
    chord_types: dict[str, list[int]],
) -> tuple[int, str]:
    """Get root and chord type from class index.

    Args:
        class_idx: Class index
        chord_types: Dictionary of chord types

    Returns:
        Tuple of (root, chord_type)
    """
    num_types = len(chord_types)
    root = class_idx // num_types
    type_idx = class_idx % num_types
    chord_type = list(chord_types.keys())[type_idx]
    return root, chord_type


def notes_to_pitch_classes(notes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Convert MIDI notes to pitch classes (0-11).

    Args:
        notes: MIDI note numbers

    Returns:
        Pitch classes (mod 12)
    """
    return tuple(sorted(set(n % 12 for n in notes)))
