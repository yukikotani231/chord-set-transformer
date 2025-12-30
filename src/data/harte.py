"""Harte notation parser for chord symbols.

Harte notation format: ROOT:QUALITY(/BASS)
Examples:
  - C:maj     -> C major
  - G:min7    -> G minor 7th
  - D:7       -> D dominant 7th
  - F:maj/5   -> F major with 5th in bass (2nd inversion)
  - B:dim/b3  -> B diminished with b3 in bass
"""

import re
from dataclasses import dataclass

# Root note to pitch class mapping
ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0,
}

# Chord quality to intervals (semitones from root)
QUALITY_TO_INTERVALS = {
    # Triads
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    # Seventh chords
    "7": [0, 4, 7, 10],  # Dominant 7th
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],  # Half-diminished
    "minmaj7": [0, 3, 7, 11],
    "aug7": [0, 4, 8, 10],
    # Extended chords (simplified)
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    # Power chord
    "5": [0, 7],
    # No chord
    "N": [],
}

# Mapping from Harte shorthand to our quality names
HARTE_QUALITY_MAP = {
    "": "maj",  # No quality = major
    "maj": "maj",
    "min": "min",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "7": "7",
    "maj7": "maj7",
    "min7": "min7",
    "dim7": "dim7",
    "hdim7": "hdim7",
    "minmaj7": "minmaj7",
    "aug7": "aug7",
    "9": "9",
    "maj9": "maj9",
    "min9": "min9",
    "5": "5",
    "(1)": "5",  # Just root
    "1": "5",
}


@dataclass
class ParsedChord:
    """Parsed chord from Harte notation."""

    root: int  # Pitch class (0-11)
    quality: str  # Quality name
    bass: int | None  # Bass pitch class if specified
    original: str  # Original string

    @property
    def intervals(self) -> list[int]:
        """Get intervals for this chord quality."""
        return QUALITY_TO_INTERVALS.get(self.quality, [])

    def to_midi_notes(self, octave: int = 4) -> list[int]:
        """Convert to MIDI note numbers.

        Args:
            octave: Base octave (default 4, so C4=60)

        Returns:
            List of MIDI note numbers.
        """
        if not self.intervals:
            return []

        base = 12 * (octave + 1) + self.root
        notes = [base + interval for interval in self.intervals]

        # Handle bass note (inversion)
        if self.bass is not None:
            # Find which note matches the bass and move it down
            bass_pc = self.bass
            for i, note in enumerate(notes):
                if note % 12 == bass_pc:
                    # Move this note down an octave to make it the bass
                    notes[i] -= 12
                    break
            else:
                # Bass note not in chord, add it
                bass_note = base + (bass_pc - self.root) % 12 - 12
                notes.insert(0, bass_note)

        return sorted(notes)

    @property
    def simplified_quality(self) -> str:
        """Get simplified quality for classification.

        Reduces to basic categories: maj, min, dim, aug, dom7, maj7, min7, other
        """
        if self.quality in ["maj", "sus2", "sus4", "5"]:
            return "maj"
        elif self.quality == "min":
            return "min"
        elif self.quality in ["dim", "dim7", "hdim7"]:
            return "dim"
        elif self.quality in ["aug", "aug7"]:
            return "aug"
        elif self.quality == "7":
            return "dom7"
        elif self.quality in ["maj7", "maj9"]:
            return "maj7"
        elif self.quality in ["min7", "min9", "minmaj7"]:
            return "min7"
        else:
            return "other"


def parse_harte(chord_str: str) -> ParsedChord | None:
    """Parse a Harte notation chord string.

    Args:
        chord_str: Chord string like "C:maj7/5" or "N" for no chord

    Returns:
        ParsedChord or None if parsing fails
    """
    chord_str = chord_str.strip()

    # Handle special cases
    if chord_str in ["N", "X", ""]:
        return None

    # Pattern: ROOT:QUALITY(/BASS)
    # ROOT: letter + optional accidental
    # QUALITY: optional quality string
    # BASS: optional /bass note
    pattern = r"^([A-Ga-g][#b]?):?([^/]*)(?:/(.+))?$"
    match = re.match(pattern, chord_str)

    if not match:
        return None

    root_str, quality_str, bass_str = match.groups()

    # Parse root
    root_str = root_str[0].upper() + root_str[1:] if len(root_str) > 1 else root_str.upper()
    if root_str not in ROOT_TO_PC:
        return None
    root = ROOT_TO_PC[root_str]

    # Parse quality
    quality_str = quality_str.strip() if quality_str else ""

    # Handle complex qualities by simplifying
    # e.g., "maj7(9)" -> "maj7", "min(*3,*5)" -> "min"
    quality_str = re.sub(r"\([^)]*\)", "", quality_str)
    quality_str = quality_str.strip("()")

    quality = HARTE_QUALITY_MAP.get(quality_str, quality_str)
    if quality not in QUALITY_TO_INTERVALS:
        # Try to match partial
        for q in ["maj7", "min7", "dim7", "7", "maj", "min", "dim", "aug"]:
            if q in quality_str.lower():
                quality = q
                break
        else:
            quality = "maj"  # Default to major

    # Parse bass
    bass = None
    if bass_str:
        # Bass can be a scale degree (1, 3, 5, b3, #5, etc.) or a note name
        bass_str = bass_str.strip()
        if bass_str in ROOT_TO_PC:
            bass = ROOT_TO_PC[bass_str]
        else:
            # Parse scale degree
            degree_map = {
                "1": 0,
                "b2": 1,
                "2": 2,
                "b3": 3,
                "3": 4,
                "4": 5,
                "#4": 6,
                "b5": 6,
                "5": 7,
                "#5": 8,
                "b6": 8,
                "6": 9,
                "b7": 10,
                "7": 11,
            }
            if bass_str in degree_map:
                bass = (root + degree_map[bass_str]) % 12

    return ParsedChord(
        root=root,
        quality=quality,
        bass=bass,
        original=chord_str,
    )


def get_unique_qualities(chord_strings: list[str]) -> set[str]:
    """Get unique chord qualities from a list of chord strings."""
    qualities = set()
    for cs in chord_strings:
        parsed = parse_harte(cs)
        if parsed:
            qualities.add(parsed.quality)
    return qualities
