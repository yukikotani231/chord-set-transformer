"""Tests for model components."""

import torch

from src.models.classifier import ChordSetTransformer, SimpleChordClassifier
from src.models.modules import ISAB, MAB, PMA, SAB


class TestMAB:
    def test_forward(self):
        mab = MAB(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        y = torch.randn(2, 10, 64)
        out = mab(x, y)
        assert out.shape == (2, 10, 64)

    def test_with_mask(self):
        mab = MAB(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        y = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 5:] = False
        out = mab(x, y, mask)
        assert out.shape == (2, 10, 64)


class TestSAB:
    def test_forward(self):
        sab = SAB(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = sab(x)
        assert out.shape == (2, 10, 64)

    def test_permutation_equivariance(self):
        sab = SAB(d_model=64, num_heads=4)
        sab.eval()
        x = torch.randn(1, 5, 64)

        # Original output
        with torch.no_grad():
            out1 = sab(x)

        # Permuted input
        perm = torch.tensor([2, 0, 4, 1, 3])
        x_perm = x[:, perm, :]
        with torch.no_grad():
            out2 = sab(x_perm)

        # Output should be permuted accordingly
        out1_perm = out1[:, perm, :]
        assert torch.allclose(out1_perm, out2, atol=1e-5)


class TestISAB:
    def test_forward(self):
        isab = ISAB(d_model=64, num_heads=4, num_inducing_points=8)
        x = torch.randn(2, 10, 64)
        out = isab(x)
        assert out.shape == (2, 10, 64)


class TestPMA:
    def test_forward(self):
        pma = PMA(d_model=64, num_heads=4, num_seeds=1)
        x = torch.randn(2, 10, 64)
        out = pma(x)
        assert out.shape == (2, 1, 64)

    def test_multiple_seeds(self):
        pma = PMA(d_model=64, num_heads=4, num_seeds=4)
        x = torch.randn(2, 10, 64)
        out = pma(x)
        assert out.shape == (2, 4, 64)


class TestChordSetTransformer:
    def test_forward(self):
        model = ChordSetTransformer(
            input_dim=14,
            num_classes=48,
            d_model=64,
            num_heads=4,
            num_layers=2,
        )
        x = torch.randn(4, 5, 14)
        mask = torch.ones(4, 5, dtype=torch.bool)
        out = model(x, mask)
        assert out["logits"].shape == (4, 48)
        assert out["pooled"].shape == (4, 64)

    def test_variable_length(self):
        model = ChordSetTransformer(input_dim=14, num_classes=8)
        x = torch.randn(2, 10, 14)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[1, 5:] = False
        out = model(x, mask)
        assert out["logits"].shape == (2, 8)

    def test_with_isab(self):
        model = ChordSetTransformer(
            input_dim=14,
            num_classes=8,
            use_isab=True,
            num_inducing_points=4,
        )
        x = torch.randn(2, 10, 14)
        out = model(x)
        assert out["logits"].shape == (2, 8)


class TestSimpleChordClassifier:
    def test_forward(self):
        model = SimpleChordClassifier(input_dim=14, num_classes=48)
        x = torch.randn(4, 5, 14)
        mask = torch.ones(4, 5, dtype=torch.bool)
        out = model(x, mask)
        assert out["logits"].shape == (4, 48)

    def test_permutation_invariance(self):
        model = SimpleChordClassifier(input_dim=14, num_classes=8)
        model.eval()
        x = torch.randn(1, 5, 14)
        mask = torch.ones(1, 5, dtype=torch.bool)

        with torch.no_grad():
            out1 = model(x, mask)

        # Permute input
        perm = torch.tensor([2, 0, 4, 1, 3])
        x_perm = x[:, perm, :]
        with torch.no_grad():
            out2 = model(x_perm, mask)

        # Output should be the same (permutation invariant)
        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)
