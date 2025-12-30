"""Tests for datasets."""

import torch

from src.data.dataset import (
    EnhancedChordDataset,
    SyntheticChordDataset,
    collate_fn,
)


class TestSyntheticChordDataset:
    def test_creation(self):
        dataset = SyntheticChordDataset(num_samples=100)
        assert len(dataset) == 100
        assert dataset.num_classes == 48  # 12 roots * 4 types
        assert dataset.feature_dim == 1

    def test_sample_structure(self):
        dataset = SyntheticChordDataset(num_samples=10)
        sample = dataset[0]
        assert "features" in sample
        assert "label" in sample
        assert "num_notes" in sample
        assert sample["features"].dim() == 2
        assert sample["label"].dim() == 0

    def test_reproducibility(self):
        dataset1 = SyntheticChordDataset(num_samples=10, seed=42)
        dataset2 = SyntheticChordDataset(num_samples=10, seed=42)
        assert torch.allclose(dataset1[0]["features"], dataset2[0]["features"])


class TestEnhancedChordDataset:
    def test_creation(self):
        dataset = EnhancedChordDataset(num_samples=100)
        assert len(dataset) == 100
        assert dataset.feature_dim == 14

    def test_feature_structure(self):
        dataset = EnhancedChordDataset(num_samples=10)
        sample = dataset[0]
        features = sample["features"]
        # First 12 dims should be one-hot pitch class
        assert features[:, :12].sum(dim=1).allclose(torch.ones(features.shape[0]))


class TestCollateFn:
    def test_padding(self):
        dataset = EnhancedChordDataset(num_samples=10)
        samples = [dataset[i] for i in range(4)]
        batch = collate_fn(samples)

        assert "features" in batch
        assert "labels" in batch
        assert "mask" in batch
        assert batch["features"].dim() == 3
        assert batch["labels"].dim() == 1
        assert batch["mask"].dim() == 2

    def test_mask_correctness(self):
        dataset = EnhancedChordDataset(num_samples=10)
        samples = [dataset[i] for i in range(2)]
        batch = collate_fn(samples)

        for i, sample in enumerate(samples):
            n = sample["num_notes"].item()
            assert batch["mask"][i, :n].all()
            if n < batch["mask"].shape[1]:
                assert not batch["mask"][i, n:].any()
