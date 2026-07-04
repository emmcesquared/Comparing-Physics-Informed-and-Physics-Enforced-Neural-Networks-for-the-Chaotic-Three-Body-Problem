"""
Deep Neural Network for Three-Body Problem Position Prediction.

Architecture adapted from Breen et al. paper:
- Input: (t, x2_0, z2_0) - time + initial position of body 2
- Output: (x1, z1, x2, z2) - positions of bodies 1 and 2 at time t
- Body 3 inferred via center-of-mass constraint: p3 = -(p1 + p2)

Network: 10 hidden layers, 128 neurons each, ReLU activations
"""

import torch
import torch.nn as nn


class ThreeBodyDNN(nn.Module):
    """
    Feed-forward DNN for three-body position prediction.

    Input: [t, x2_0, z2_0] (3 features)
    Output: [x1, z1, x2, z2] (4 positions)

    Body 3 position reconstructed as: p3 = -(p1 + p2)
    """

    def __init__(self, hidden_dim=128, num_layers=10, dropout=0.0):
        super().__init__()

        self.input_dim = 3   # t, x2_0, z2_0
        self.output_dim = 4  # x1, z1, x2, z2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer (linear)
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 3) containing [t, x2_0, z2_0]

        Returns:
            Tensor of shape (batch, 4) containing [x1, z1, x2, z2]
        """
        return self.network(x)

    def predict_all_bodies(self, x):
        """
        Predict positions for all 3 bodies.

        Args:
            x: Tensor of shape (batch, 3) containing [t, x2_0, z2_0]

        Returns:
            Tensor of shape (batch, 6) containing [x1, z1, x2, z2, x3, z3]
        """
        out = self.forward(x)  # [x1, z1, x2, z2]

        # Body 3 from COM constraint: p3 = -(p1 + p2)
        x1, z1 = out[:, 0], out[:, 1]
        x2, z2 = out[:, 2], out[:, 3]
        x3 = -(x1 + x2)
        z3 = -(z1 + z2)

        return torch.stack([x1, z1, x2, z2, x3, z3], dim=1)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ThreeBodyDNNLarge(ThreeBodyDNN):
    """Larger variant with 256 neurons and 12 layers."""
    def __init__(self, dropout=0.0):
        super().__init__(hidden_dim=256, num_layers=12, dropout=dropout)


class ThreeBodyDNNSmall(ThreeBodyDNN):
    """Smaller variant with 64 neurons and 6 layers."""
    def __init__(self, dropout=0.0):
        super().__init__(hidden_dim=64, num_layers=6, dropout=dropout)


if __name__ == "__main__":
    # Test the model
    model = ThreeBodyDNN()
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch = torch.randn(32, 3)  # [t, x2_0, z2_0]
    out = model(batch)
    print(f"\nInput shape: {batch.shape}")
    print(f"Output shape: {out.shape}")

    # Test full prediction
    full_out = model.predict_all_bodies(batch)
    print(f"Full output shape: {full_out.shape}")
