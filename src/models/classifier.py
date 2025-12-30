"""Chord classifier using Set Transformer."""

import torch.nn as nn
from torch import Tensor

from .modules import ISAB, PMA, SAB


class ChordSetTransformer(nn.Module):
    """Set Transformer for chord classification.

    Architecture:
    1. Input projection: note features -> d_model
    2. Set encoder: stack of SAB or ISAB layers
    3. Pooling: PMA to aggregate set into fixed-size representation
    4. Classifier: MLP to predict chord class
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_isab: bool = False,
        num_inducing_points: int = 8,
    ):
        """Initialize chord classifier.

        Args:
            input_dim: Dimension of input features per note.
            num_classes: Number of chord classes.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of encoder layers.
            dropout: Dropout probability.
            use_isab: Whether to use ISAB (for larger sets).
            num_inducing_points: Number of inducing points for ISAB.
        """
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Build encoder layers
        encoder_layers: list[SAB | ISAB] = []
        for _ in range(num_layers):
            if use_isab:
                encoder_layers.append(ISAB(d_model, num_heads, num_inducing_points, dropout))
            else:
                encoder_layers.append(SAB(d_model, num_heads, dropout))
        self.encoder = nn.ModuleList(encoder_layers)

        # Pooling
        self.pooling = PMA(d_model, num_heads, num_seeds=1, dropout=dropout)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Note features of shape (batch, num_notes, input_dim)
            mask: Valid note mask of shape (batch, num_notes)

        Returns:
            Dictionary with 'logits' of shape (batch, num_classes)
        """
        # Project input
        h = self.input_proj(x)

        # Encode set
        for layer in self.encoder:
            h = layer(h, mask)

        # Pool to single vector
        pooled = self.pooling(h, mask)  # (batch, 1, d_model)
        pooled = pooled.squeeze(1)  # (batch, d_model)

        # Classify
        logits = self.classifier(pooled)

        return {"logits": logits, "pooled": pooled}


class SimpleChordClassifier(nn.Module):
    """Simple baseline: DeepSets-style classifier.

    Uses element-wise MLP + sum pooling.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Encode each note
        h = self.encoder(x)  # (batch, num_notes, hidden_dim)

        # Masked sum pooling
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()

        pooled = h.sum(dim=1)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(pooled)

        return {"logits": logits, "pooled": pooled}


def create_chord_classifier(
    input_dim: int,
    num_classes: int,
    model_type: str = "set_transformer",
    **kwargs,
) -> nn.Module:
    """Factory function to create chord classifier.

    Args:
        input_dim: Input feature dimension.
        num_classes: Number of chord classes.
        model_type: 'set_transformer' or 'deepsets'
        **kwargs: Additional model arguments.

    Returns:
        Chord classifier model.
    """
    if model_type == "set_transformer":
        return ChordSetTransformer(input_dim, num_classes, **kwargs)
    elif model_type == "deepsets":
        return SimpleChordClassifier(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
