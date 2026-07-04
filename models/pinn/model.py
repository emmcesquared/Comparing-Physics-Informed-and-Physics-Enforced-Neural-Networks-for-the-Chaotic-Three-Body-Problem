"""
Physics-Informed Neural Network with ResNet Architecture for Three-Body Problem.

Architecture from Breen et al. paper:
- Input: (t, x2_0, z2_0) - time + initial position of body 2
- Output: (x1, z1, x2, z2) - positions of bodies 1 and 2 at time t
- Body 3 inferred via center-of-mass constraint: p3 = -(p1 + p2)

Network: ResNet with skip connections, Tanh activations
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim, activation=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = activation()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + residual)  # Skip connection


class ThreeBodyPINN(nn.Module):
    """
    Physics-Informed Neural Network with ResNet architecture.

    Input: [t, x2_0, z2_0] (3 features)
    Output: [x1, z1, x2, z2] (4 positions)

    Body 3 position reconstructed as: p3 = -(p1 + p2)
    """

    def __init__(self, hidden_dim=128, num_blocks=5, activation=nn.Tanh):
        super().__init__()

        self.input_dim = 3   # t, x2_0, z2_0
        self.output_dim = 4  # x1, z1, x2, z2
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection
        self.input_layer = nn.Linear(self.input_dim, hidden_dim)
        self.input_act = activation()

        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, activation) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
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
        # Input projection
        h = self.input_act(self.input_layer(x))

        # ResNet blocks
        for block in self.res_blocks:
            h = block(h)

        # Output
        return self.output_layer(h)

    def forward_with_grad(self, t, ic):
        """
        Forward pass that preserves gradients for physics loss.

        Args:
            t: Tensor of shape (batch, 1) - time (requires_grad=True)
            ic: Tensor of shape (batch, 2) - initial conditions [x2_0, z2_0]

        Returns:
            positions: (batch, 4) - [x1, z1, x2, z2]
        """
        x = torch.cat([t, ic], dim=1)
        return self.forward(x)

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


class ThreeBodyPINNLarge(ThreeBodyPINN):
    """Larger variant with 256 neurons and 8 blocks."""
    def __init__(self):
        super().__init__(hidden_dim=256, num_blocks=8)


class ThreeBodyPINNSmall(ThreeBodyPINN):
    """Smaller variant with 64 neurons and 3 blocks."""
    def __init__(self):
        super().__init__(hidden_dim=64, num_blocks=3)


def gravitational_acceleration(positions, G=1.0, masses=None):
    """
    Compute gravitational acceleration for each body.

    Args:
        positions: (batch, 4) tensor [x1, z1, x2, z2]
        G: gravitational constant
        masses: list of 3 masses (default [1,1,1])

    Returns:
        accelerations: (batch, 4) tensor [ax1, az1, ax2, az2]
    """
    if masses is None:
        masses = torch.tensor([1.0, 1.0, 1.0], device=positions.device)

    batch_size = positions.shape[0]

    # Extract positions
    x1, z1 = positions[:, 0], positions[:, 1]
    x2, z2 = positions[:, 2], positions[:, 3]
    # Body 3 from COM constraint
    x3 = -(x1 + x2)
    z3 = -(z1 + z2)

    # Position vectors for all bodies: (batch, 3, 2)
    pos = torch.stack([
        torch.stack([x1, z1], dim=1),
        torch.stack([x2, z2], dim=1),
        torch.stack([x3, z3], dim=1)
    ], dim=1)

    # Compute accelerations
    acc = torch.zeros(batch_size, 3, 2, device=positions.device)

    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = pos[:, j] - pos[:, i]  # (batch, 2)
                r_norm = torch.norm(r_ij, dim=1, keepdim=True) + 1e-10
                acc[:, i] += G * masses[j] * r_ij / (r_norm ** 3)

    # Return accelerations for bodies 1 and 2 only
    return torch.cat([acc[:, 0], acc[:, 1]], dim=1)  # (batch, 4)


if __name__ == "__main__":
    # Test the model
    model = ThreeBodyPINN()
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

    # Test gravitational acceleration
    pos = torch.randn(32, 4)
    acc = gravitational_acceleration(pos)
    print(f"Acceleration shape: {acc.shape}")
