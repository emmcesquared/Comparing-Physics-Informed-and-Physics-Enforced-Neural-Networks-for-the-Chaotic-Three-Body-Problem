"""
Hamiltonian Neural Network (HNN) for Three-Body Problem.

SAME INTERFACE AS DNN/PINN/LNN:
- Input: (t, x2_0, z2_0) - time + initial conditions
- Output: (x1, z1, x2, z2) - positions at time t

Internal physics: Learns Hamiltonian H(q, p) and enforces Hamilton's equations
as a physics constraint during training.

Architecture:
1. Position Network: Maps (t, IC) -> positions (like DNN)
2. Hamiltonian Network: Learns H(q, p) for physics constraint
3. Hamilton's equations residual: dq/dt = dH/dp, dp/dt = -dH/dq

The key insight: We predict positions directly (for fair comparison with DNN/PINN/LNN),
but regularize the predictions to satisfy Hamiltonian mechanics.

Hamilton's equations:
    dq_i/dt =  dH/dp_i
    dp_i/dt = -dH/dq_i

For equal masses (m=1), momentum p = v, so:
    q = [x1, z1, x2, z2]
    p = [vx1, vz1, vx2, vz2]

Hamiltonian vector-field residual (from metrics spec):
    R_H(t) = ||J * grad(H) - s_dot||_2
    where J = [[0, I], [-I, 0]] is the symplectic matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HamiltonianNetwork(nn.Module):
    """
    Neural network that learns the Hamiltonian H(q, p).

    Input: [q, p] = [x1,z1,x2,z2, px1,pz1,px2,pz2] (8D)
    Output: scalar H (Hamiltonian / total energy value)
    """

    def __init__(self, hidden_dim=128, num_layers=4, activation='softplus'):
        super().__init__()

        self.input_dim = 8   # 4 positions + 4 momenta
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q, p):
        """
        Compute Hamiltonian H(q, p).

        Args:
            q: (batch, 4) positions [x1,z1,x2,z2]
            p: (batch, 4) momenta [px1,pz1,px2,pz2]

        Returns:
            H: (batch, 1) Hamiltonian values
        """
        x = torch.cat([q, p], dim=1)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.output_layer(x)


class SeparableHamiltonianNetwork(nn.Module):
    """
    Separable Hamiltonian network that explicitly learns H = T(p) + V(q).

    This structure enforces the physics prior that the Hamiltonian
    separates into kinetic energy T(p) and potential energy V(q).

    Physics prior:
    - T(p) depends only on momenta (kinetic energy)
    - V(q) depends only on positions (potential energy)
    - H = T + V (Hamiltonian structure)
    """

    def __init__(self, hidden_dim=64, num_layers=3, activation='softplus'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        # Kinetic energy network T(p): 4D -> 1D
        t_layers = []
        t_layers.append(nn.Linear(4, hidden_dim))
        for _ in range(num_layers - 1):
            t_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.t_layers = nn.ModuleList(t_layers)
        self.t_output = nn.Linear(hidden_dim, 1)

        # Potential energy network V(q): 4D -> 1D
        v_layers = []
        v_layers.append(nn.Linear(4, hidden_dim))
        for _ in range(num_layers - 1):
            v_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.v_layers = nn.ModuleList(v_layers)
        self.v_output = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def kinetic_energy(self, p):
        """Compute learned kinetic energy T(p)."""
        x = p
        for layer in self.t_layers:
            x = self.act(layer(x))
        # Use softplus on output to ensure T >= 0
        return F.softplus(self.t_output(x))

    def potential_energy(self, q):
        """Compute learned potential energy V(q)."""
        x = q
        for layer in self.v_layers:
            x = self.act(layer(x))
        return self.v_output(x)

    def forward(self, q, p):
        """
        Compute Hamiltonian H = T(p) + V(q).

        Args:
            q: (batch, 4) positions
            p: (batch, 4) momenta

        Returns:
            H: (batch, 1) Hamiltonian values
        """
        T = self.kinetic_energy(p)
        V = self.potential_energy(q)
        return T + V


class ThreeBodyHNN(nn.Module):
    """
    Hamiltonian Neural Network for three-body position prediction.

    SAME INTERFACE AS DNN/PINN/LNN:
    - Input: [t, x2_0, z2_0] (3 features)
    - Output: [x1, z1, x2, z2] (4 positions)

    Internal: Uses Hamiltonian network for physics-informed regularization
    via Hamilton's equations.

    Args:
        use_separable_hamiltonian: If True, use SeparableHamiltonianNetwork with
            explicit T+V structure. Recommended to prevent trivial solutions.
    """

    def __init__(self, hidden_dim=128, num_layers=4, activation='softplus',
                 hamiltonian_hidden=64, hamiltonian_layers=3,
                 use_separable_hamiltonian=True):
        super().__init__()

        self.input_dim = 3   # t, x2_0, z2_0
        self.output_dim = 4  # x1, z1, x2, z2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_separable_hamiltonian = use_separable_hamiltonian

        if activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        # Position prediction network (like DNN, but with smooth activations)
        pos_layers = []
        pos_layers.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            pos_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.pos_layers = nn.ModuleList(pos_layers)
        self.pos_output = nn.Linear(hidden_dim, self.output_dim)

        # Hamiltonian network for physics constraint
        if use_separable_hamiltonian:
            self.hamiltonian_net = SeparableHamiltonianNetwork(
                hidden_dim=hamiltonian_hidden,
                num_layers=hamiltonian_layers,
                activation=activation
            )
        else:
            self.hamiltonian_net = HamiltonianNetwork(
                hidden_dim=hamiltonian_hidden,
                num_layers=hamiltonian_layers,
                activation=activation
            )

        self._init_weights()

    def _init_weights(self):
        for m in [*self.pos_layers, self.pos_output]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass - predict positions.

        Args:
            x: Tensor of shape (batch, 3) containing [t, x2_0, z2_0]

        Returns:
            Tensor of shape (batch, 4) containing [x1, z1, x2, z2]
        """
        h = x
        for layer in self.pos_layers:
            h = self.act(layer(h))
        return self.pos_output(h)

    def forward_with_physics(self, t, ic, return_derivatives=False):
        """
        Forward pass that also computes physics quantities for Hamilton's equations loss.

        Args:
            t: (batch, 1) time values (requires_grad should be True)
            ic: (batch, 2) initial conditions [x2_0, z2_0]
            return_derivatives: if True, return velocities and accelerations

        Returns:
            positions: (batch, 4)
            If return_derivatives:
                velocities: (batch, 4)
                accelerations: (batch, 4)
        """
        x = torch.cat([t, ic], dim=1)  # (batch, 3)
        positions = self.forward(x)     # (batch, 4)

        if not return_derivatives:
            return positions

        # Compute velocities via autodiff: dq/dt
        ones = torch.ones_like(positions[:, 0])
        velocities = []
        for i in range(4):
            grad_i = torch.autograd.grad(
                positions[:, i], t,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True
            )[0]
            velocities.append(grad_i)
        velocities = torch.cat(velocities, dim=1)  # (batch, 4)

        # Compute accelerations via autodiff: d2q/dt2
        accelerations = []
        for i in range(4):
            grad_i = torch.autograd.grad(
                velocities[:, i], t,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True
            )[0]
            accelerations.append(grad_i)
        accelerations = torch.cat(accelerations, dim=1)  # (batch, 4)

        return positions, velocities, accelerations

    def hamiltonian(self, q, p):
        """Compute Hamiltonian H(q, p)."""
        return self.hamiltonian_net(q, p)

    def compute_hamiltonian_vector_field_residual(self, q, p):
        """
        Compute the Hamiltonian vector field: J * grad(H).

        Hamilton's equations:
            dq/dt =  dH/dp
            dp/dt = -dH/dq

        This is equivalent to: s_dot = J * grad_H(s)
        where s = [q, p] and J = [[0, I], [-I, 0]]

        Args:
            q: (batch, 4) positions (requires_grad=True)
            p: (batch, 4) momenta (requires_grad=True)

        Returns:
            dq_dt_H: (batch, 4) = dH/dp (predicted velocity from Hamiltonian)
            dp_dt_H: (batch, 4) = -dH/dq (predicted force from Hamiltonian)
        """
        # Compute Hamiltonian
        H = self.hamiltonian(q, p)  # (batch, 1)

        # dH/dq
        dH_dq = torch.autograd.grad(
            H.sum(), q, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if dH_dq is None:
            dH_dq = torch.zeros_like(q)

        # dH/dp
        dH_dp = torch.autograd.grad(
            H.sum(), p, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if dH_dp is None:
            dH_dp = torch.zeros_like(p)

        # Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq
        dq_dt_H = dH_dp
        dp_dt_H = -dH_dq

        return dq_dt_H, dp_dt_H

    def predict_all_bodies(self, x):
        """
        Predict positions for all 3 bodies.

        Args:
            x: Tensor of shape (batch, 3) containing [t, x2_0, z2_0]

        Returns:
            Tensor of shape (batch, 6) containing [x1, z1, x2, z2, x3, z3]
        """
        out = self.forward(x)  # [x1, z1, x2, z2]

        x1, z1 = out[:, 0], out[:, 1]
        x2, z2 = out[:, 2], out[:, 3]
        x3 = -(x1 + x2)
        z3 = -(z1 + z2)

        return torch.stack([x1, z1, x2, z2, x3, z3], dim=1)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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

    x1, z1 = positions[:, 0], positions[:, 1]
    x2, z2 = positions[:, 2], positions[:, 3]
    x3 = -(x1 + x2)
    z3 = -(z1 + z2)

    pos = torch.stack([
        torch.stack([x1, z1], dim=1),
        torch.stack([x2, z2], dim=1),
        torch.stack([x3, z3], dim=1)
    ], dim=1)

    acc = torch.zeros(batch_size, 3, 2, device=positions.device)

    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = pos[:, j] - pos[:, i]
                r_norm = torch.norm(r_ij, dim=1, keepdim=True) + 1e-10
                acc[:, i] += G * masses[j] * r_ij / (r_norm ** 3)

    return torch.cat([acc[:, 0], acc[:, 1]], dim=1)


class ThreeBodyHNNLarge(ThreeBodyHNN):
    """Larger variant."""
    def __init__(self, activation='softplus'):
        super().__init__(hidden_dim=256, num_layers=8, activation=activation,
                         hamiltonian_hidden=128, hamiltonian_layers=4)


class ThreeBodyHNNSmall(ThreeBodyHNN):
    """Smaller variant."""
    def __init__(self, activation='softplus'):
        super().__init__(hidden_dim=64, num_layers=4, activation=activation,
                         hamiltonian_hidden=32, hamiltonian_layers=2)


if __name__ == "__main__":
    print("Testing ThreeBodyHNN (DNN/PINN/LNN-compatible interface)...")
    model = ThreeBodyHNN(hidden_dim=128, num_layers=4)
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass (same as DNN/PINN/LNN)
    batch = torch.randn(32, 3)  # [t, x2_0, z2_0]
    out = model(batch)
    print(f"\nInput shape: {batch.shape}")
    print(f"Output shape: {out.shape}")

    # Test full prediction
    full_out = model.predict_all_bodies(batch)
    print(f"Full output shape: {full_out.shape}")

    # Test physics-aware forward
    print("\nTesting forward_with_physics...")
    t = torch.randn(32, 1, requires_grad=True)
    ic = torch.randn(32, 2)
    pos, vel, acc = model.forward_with_physics(t, ic, return_derivatives=True)
    print(f"Positions: {pos.shape}, Velocities: {vel.shape}, Accelerations: {acc.shape}")

    # Test Hamiltonian vector field
    print("\nTesting Hamiltonian vector field residual...")
    q = torch.randn(8, 4, requires_grad=True)
    p = torch.randn(8, 4, requires_grad=True)
    dq_dt_H, dp_dt_H = model.compute_hamiltonian_vector_field_residual(q, p)
    print(f"dq/dt from H: {dq_dt_H.shape}, dp/dt from H: {dp_dt_H.shape}")

    # Test Hamiltonian value
    H = model.hamiltonian(q, p)
    print(f"Hamiltonian shape: {H.shape}")

    # Test gravitational acceleration
    print("\nTesting gravitational acceleration...")
    pos = torch.randn(32, 4)
    acc = gravitational_acceleration(pos)
    print(f"Gravitational acceleration shape: {acc.shape}")
