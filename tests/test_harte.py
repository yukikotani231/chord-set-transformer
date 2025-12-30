"""Tests for Harte notation parser."""

from src.data.harte import parse_harte


class TestParseHarte:
    def test_simple_major(self):
        chord = parse_harte("C:maj")
        assert chord is not None
        assert chord.root == 0
        assert chord.quality == "maj"
        assert chord.bass is None

    def test_simple_minor(self):
        chord = parse_harte("A:min")
        assert chord is not None
        assert chord.root == 9
        assert chord.quality == "min"

    def test_dominant_seventh(self):
        chord = parse_harte("G:7")
        assert chord is not None
        assert chord.root == 7
        assert chord.quality == "7"

    def test_major_seventh(self):
        chord = parse_harte("F:maj7")
        assert chord is not None
        assert chord.root == 5
        assert chord.quality == "maj7"

    def test_with_bass_note(self):
        chord = parse_harte("C:maj/E")
        assert chord is not None
        assert chord.root == 0
        assert chord.quality == "maj"
        assert chord.bass == 4  # E

    def test_with_bass_degree(self):
        chord = parse_harte("C:maj/5")
        assert chord is not None
        assert chord.bass == 7  # G (5th of C)

    def test_flat_root(self):
        chord = parse_harte("Bb:min")
        assert chord is not None
        assert chord.root == 10

    def test_sharp_root(self):
        chord = parse_harte("F#:dim")
        assert chord is not None
        assert chord.root == 6

    def test_no_chord(self):
        assert parse_harte("N") is None
        assert parse_harte("X") is None
        assert parse_harte("") is None

    def test_invalid_format(self):
        assert parse_harte("invalid") is None


class TestParsedChordToMidi:
    def test_c_major(self):
        chord = parse_harte("C:maj")
        notes = chord.to_midi_notes(octave=4)
        assert notes == [60, 64, 67]

    def test_a_minor(self):
        chord = parse_harte("A:min")
        notes = chord.to_midi_notes(octave=4)
        assert notes == [69, 72, 76]

    def test_with_inversion(self):
        chord = parse_harte("C:maj/5")
        notes = chord.to_midi_notes(octave=4)
        # G should be in bass
        assert min(notes) % 12 == 7  # G

    def test_different_octave(self):
        chord = parse_harte("C:maj")
        notes_3 = chord.to_midi_notes(octave=3)
        notes_5 = chord.to_midi_notes(octave=5)
        assert notes_5[0] - notes_3[0] == 24  # 2 octaves difference


class TestSimplifiedQuality:
    def test_major_variants(self):
        assert parse_harte("C:maj").simplified_quality == "maj"
        assert parse_harte("C:sus4").simplified_quality == "maj"

    def test_minor_variants(self):
        assert parse_harte("C:min").simplified_quality == "min"

    def test_diminished_variants(self):
        assert parse_harte("C:dim").simplified_quality == "dim"
        assert parse_harte("C:dim7").simplified_quality == "dim"
        assert parse_harte("C:hdim7").simplified_quality == "dim"

    def test_seventh_variants(self):
        assert parse_harte("C:7").simplified_quality == "dom7"
        assert parse_harte("C:maj7").simplified_quality == "maj7"
        assert parse_harte("C:min7").simplified_quality == "min7"
