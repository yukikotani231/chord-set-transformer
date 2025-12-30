"""Tests for chord definitions."""

import pytest

from src.data.chords import (
    TRIAD_TYPES,
    Chord,
    get_chord_class_index,
    get_chord_from_class_index,
    notes_to_pitch_classes,
)


class TestChord:
    def test_create_c_major(self):
        chord = Chord.from_root_and_type(0, "maj", octave=4)
        assert chord.root == 0
        assert chord.chord_type == "maj"
        assert chord.notes == (60, 64, 67)  # C4, E4, G4
        assert chord.name == "C:maj"

    def test_create_a_minor(self):
        chord = Chord.from_root_and_type(9, "min", octave=4)
        assert chord.root == 9
        assert chord.chord_type == "min"
        assert chord.notes == (69, 72, 76)  # A4, C5, E5
        assert chord.name == "A:min"

    def test_create_f_sharp_dim(self):
        chord = Chord.from_root_and_type(6, "dim", octave=3)
        assert chord.root == 6
        assert chord.name == "F#:dim"

    def test_invalid_chord_type(self):
        with pytest.raises(ValueError):
            Chord.from_root_and_type(0, "invalid")


class TestChordClassIndex:
    def test_get_class_index(self):
        # C:maj should be index 0
        idx = get_chord_class_index(0, "maj", TRIAD_TYPES)
        assert idx == 0

        # C:min should be index 1
        idx = get_chord_class_index(0, "min", TRIAD_TYPES)
        assert idx == 1

        # D:maj should be index 8 (2 * 4 + 0)
        idx = get_chord_class_index(2, "maj", TRIAD_TYPES)
        assert idx == 8

    def test_round_trip(self):
        for root in range(12):
            for chord_type in TRIAD_TYPES:
                idx = get_chord_class_index(root, chord_type, TRIAD_TYPES)
                recovered_root, recovered_type = get_chord_from_class_index(idx, TRIAD_TYPES)
                assert recovered_root == root
                assert recovered_type == chord_type


class TestNotesToPitchClasses:
    def test_c_major(self):
        notes = [60, 64, 67]  # C4, E4, G4
        pcs = notes_to_pitch_classes(notes)
        assert pcs == (0, 4, 7)

    def test_with_octave_duplicates(self):
        notes = [60, 64, 67, 72]  # C4, E4, G4, C5
        pcs = notes_to_pitch_classes(notes)
        assert pcs == (0, 4, 7)  # Duplicates removed

    def test_unsorted_input(self):
        notes = [67, 60, 64]  # G4, C4, E4
        pcs = notes_to_pitch_classes(notes)
        assert pcs == (0, 4, 7)  # Sorted output
