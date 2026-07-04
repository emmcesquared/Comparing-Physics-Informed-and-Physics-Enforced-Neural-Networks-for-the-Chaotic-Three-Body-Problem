"""
Lagrangian Neural Network (LNN) for Three-Body Problem.

REDESIGNED to match DNN/PINN interface:
- Input: (t, x2_0, z2_0) - time + initial conditions
- Output: (x1, z1, x2, z2) - positions at time t

Internal physics: Learns Lagrangian L(q, q̇) and enforces Euler-Lagrange equations
as a physics constraint during training.

Architecture:
1. Position Network: Maps (t, IC) -> positions (like DNN)
2. Lagrangian Network: Learns L(q, q̇) for physics constraint
3. E-L Residual: Computed via autodiff for physics loss

The key insight: We predict positions directly (for fair comparison with DNN/PINN),
but regularize the predictions to satisfy Lagrangian mechanics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagrangianNetwork(nn.Module):
    """
    Neural network that learns the Lagrangian L(q, q̇).

    Input: [q, q̇] = [x1,z1,x2,z2, vx1,vz1,vx2,vz2] (8D)
    Output: scalar L (Lagrangian value)
    """

    def __init__(self, hidden_dim=128, num_layers=4, activation='softplus'):
        super().__init__()

        self.input_dim = 8   # 4 positions + 4 velocities
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Choose activation (softplus recommended for smooth 2nd derivatives)
        if activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        # Build network
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

    def forward(self, q, qdot):
        """
        Compute Lagrangian L(q, q̇).

        Args:
            q: (batch, 4) positions [x1,z1,x2,z2]
            qdot: (batch, 4) velocities [vx1,vz1,vx2,vz2]

        Returns:
            L: (batch, 1) Lagrangian values
        """
        x = torch.cat([q, qdot], dim=1)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.output_layer(x)


class SeparableLagrangianNetwork(nn.Module):
    """
    Separable Lagrangian network that explicitly learns L = T(qdot) - V(q).
    
    This structure prevents trivial constant solutions by forcing the network
    to learn kinetic energy T(qdot) and potential energy V(q) separately.
    
    Physics prior:
    - T(qdot) depends only on velocities (kinetic energy)
    - V(q) depends only on positions (potential energy)
    - L = T - V (Lagrangian structure)
    """

    def __init__(self, hidden_dim=64, num_layers=3, activation='softplus'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Choose activation
        if activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()
        
        # Kinetic energy network T(qdot): 4D -> 1D
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
    
    def kinetic_energy(self, qdot):
        """Compute learned kinetic energy T(qdot)."""
        x = qdot
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
    
    def forward(self, q, qdot):
        """
        Compute Lagrangian L = T(qdot) - V(q).
        
        Args:
            q: (batch, 4) positions
            qdot: (batch, 4) velocities
        
        Returns:
            L: (batch, 1) Lagrangian values
        """
        T = self.kinetic_energy(qdot)
        V = self.potential_energy(q)
        return T - V


class ThreeBodyLNN(nn.Module):
    """
    Lagrangian Neural Network for three-body position prediction.

    SAME INTERFACE AS DNN/PINN:
    - Input: [t, x2_0, z2_0] (3 features)
    - Output: [x1, z1, x2, z2] (4 positions)

    Internal: Uses Lagrangian network for physics-informed regularization.

    The position network predicts trajectories, and the Lagrangian network
    provides physics constraints via the Euler-Lagrange equations.
    
    Args:
        use_separable_lagrangian: If True, use SeparableLagrangianNetwork with
            explicit T-V structure. Recommended to prevent trivial solutions.
    """

    def __init__(self, hidden_dim=128, num_layers=4, activation='softplus',
                 lagrangian_hidden=64, lagrangian_layers=3,
                 use_separable_lagrangian=True):
        super().__init__()

        self.input_dim = 3   # t, x2_0, z2_0
        self.output_dim = 4  # x1, z1, x2, z2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_separable_lagrangian = use_separable_lagrangian

        # Choose activation
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

        # Lagrangian network for physics constraint
        if use_separable_lagrangian:
            # Separable L = T(qdot) - V(q) structure (prevents trivial solutions)
            self.lagrangian_net = SeparableLagrangianNetwork(
                hidden_dim=lagrangian_hidden,
                num_layers=lagrangian_layers,
                activation=activation
            )
        else:
            # Original unconstrained Lagrangian network
            self.lagrangian_net = LagrangianNetwork(
                hidden_dim=lagrangian_hidden,
                num_layers=lagrangian_layers,
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
        Forward pass that also computes physics quantities for E-L loss.

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
        # Build input
        x = torch.cat([t, ic], dim=1)  # (batch, 3)

        # Forward through position network
        positions = self.forward(x)  # (batch, 4)

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

        # Compute accelerations via autodiff: d²q/dt²
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

    def lagrangian(self, q, qdot):
        """Compute Lagrangian L(q, q̇)."""
        return self.lagrangian_net(q, qdot)

    def compute_el_residual(self, q, qdot, qddot):
        """
        Compute Euler-Lagrange residual.

        E-L equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0

        This expands to:
            (∂²L/∂q̇²) q̈ + (∂²L/∂q∂q̇) q̇ = ∂L/∂q

        Residual = LHS - RHS

        Args:
            q: (batch, 4) positions (requires_grad=True)
            qdot: (batch, 4) velocities (requires_grad=True)
            qddot: (batch, 4) accelerations

        Returns:
            residual: (batch, 4) E-L residual for each DOF
        """
        batch_size = q.shape[0]
        n_dof = 4

        # Compute Lagrangian
        L = self.lagrangian(q, qdot)  # (batch, 1)

        # ∂L/∂q
        dL_dq = torch.autograd.grad(
            L.sum(), q, create_graph=True, retain_graph=True, allow_unused=True
        )[0]  # (batch, 4)
        if dL_dq is None:
            dL_dq = torch.zeros_like(q)

        # ∂L/∂q̇
        dL_dqdot = torch.autograd.grad(
            L.sum(), qdot, create_graph=True, retain_graph=True, allow_unused=True
        )[0]  # (batch, 4)
        if dL_dqdot is None:
            dL_dqdot = torch.zeros_like(qdot)

        # ∂²L/∂q̇² (mass matrix)
        H_qdot_qdot = []
        for i in range(n_dof):
            grad_i = torch.autograd.grad(
                dL_dqdot[:, i].sum(), qdot, create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if grad_i is None:
                grad_i = torch.zeros_like(qdot)
            H_qdot_qdot.append(grad_i)
        H_qdot_qdot = torch.stack(H_qdot_qdot, dim=1)  # (batch, 4, 4)

        # ∂²L/∂q∂q̇ (mixed Hessian - zero for separable L = T(qdot) - V(q))
        H_q_qdot = []
        for i in range(n_dof):
            grad_i = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q, create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if grad_i is None:
                grad_i = torch.zeros_like(q)
            H_q_qdot.append(grad_i)
        H_q_qdot = torch.stack(H_q_qdot, dim=1)  # (batch, 4, 4)

        # LHS: (∂²L/∂q̇²) q̈ + (∂²L/∂q∂q̇) q̇
        term1 = torch.einsum('bij,bj->bi', H_qdot_qdot, qddot)  # (batch, 4)
        term2 = torch.einsum('bij,bj->bi', H_q_qdot, qdot)      # (batch, 4)
        lhs = term1 + term2

        # RHS: ∂L/∂q
        rhs = dL_dq

        # Residual
        residual = lhs - rhs

        return residual

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


class ThreeBodyLNNLarge(ThreeBodyLNN):
    """Larger variant."""
    def __init__(self, activation='softplus'):
        super().__init__(hidden_dim=256, num_layers=8, activation=activation,
                         lagrangian_hidden=128, lagrangian_layers=4)


class ThreeBodyLNNSmall(ThreeBodyLNN):
    """Smaller variant."""
    def __init__(self, activation='softplus'):
        super().__init__(hidden_dim=64, num_layers=4, activation=activation,
                         lagrangian_hidden=32, lagrangian_layers=2)


if __name__ == "__main__":
    # Test the model
    print("Testing ThreeBodyLNN (DNN/PINN-compatible interface)...")
    model = ThreeBodyLNN(hidden_dim=128, num_layers=4)
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass (same as DNN/PINN)
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

    # Test E-L residual
    print("\nTesting E-L residual computation...")
    q = torch.randn(8, 4, requires_grad=True)
    qdot = torch.randn(8, 4, requires_grad=True)
    qddot = torch.randn(8, 4)
    residual = model.compute_el_residual(q, qdot, qddot)
    print(f"E-L Residual shape: {residual.shape}")

    # Test gravitational acceleration
    print("\nTesting gravitational acceleration...")
    pos = torch.randn(32, 4)
    acc = gravitational_acceleration(pos)
    print(f"Gravitational acceleration shape: {acc.shape}")
