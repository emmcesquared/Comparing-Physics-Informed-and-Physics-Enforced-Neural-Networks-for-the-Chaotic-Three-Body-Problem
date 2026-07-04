"""
Evaluation Metrics for LNN on Three-Body Problem.

Tier A (Common to all models - imported from DNN):
- MAE / RMSE on state
- Relative rollout error
- Energy drift
- Long-horizon error
- ODE residual

Tier B (LNN-Specific):
- Euler-Lagrange Residual: measures if predicted trajectory satisfies E-L equations
  given the learned Lagrangian
- Lagrangian Structure Score: measures if learned L has proper T - V structure
"""

import importlib.util
from pathlib import Path
import numpy as np
import torch

# Import all metrics from DNN using explicit file loading
# This avoids circular import issues
_dnn_metrics_path = Path(__file__).parent.parent / 'dnn' / 'metrics.py'
_spec = importlib.util.spec_from_file_location("dnn_metrics", _dnn_metrics_path)
_dnn_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dnn_metrics)

# Re-export everything from DNN metrics (Tier A)
compute_energy = _dnn_metrics.compute_energy
mae_state = _dnn_metrics.mae_state
rmse_state = _dnn_metrics.rmse_state
relative_error_per_step = _dnn_metrics.relative_error_per_step
rollout_error_avg = _dnn_metrics.rollout_error_avg
absolute_energy_error = _dnn_metrics.absolute_energy_error
energy_drift = _dnn_metrics.energy_drift
long_horizon_error = _dnn_metrics.long_horizon_error
compute_true_acceleration = _dnn_metrics.compute_true_acceleration
numerical_acceleration = _dnn_metrics.numerical_acceleration
ode_residual = _dnn_metrics.ode_residual
evaluate_trajectory = _dnn_metrics.evaluate_trajectory
evaluate_trajectory_with_ode = _dnn_metrics.evaluate_trajectory_with_ode
aggregate_metrics = _dnn_metrics.aggregate_metrics
print_metrics = _dnn_metrics.print_metrics


# =============================================================================
# Tier B: LNN-Specific Metrics
# =============================================================================

def compute_el_residual_from_trajectory(model, pred_pos, t, device='cuda'):
    """
    Compute Euler-Lagrange residual for a predicted trajectory.
    
    Given predicted positions at each timestep, compute velocities and
    accelerations via finite differences, then check if the learned
    Lagrangian satisfies the E-L equations.
    
    E-L equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
    
    Args:
        model: Trained LNN model with lagrangian() and compute_el_residual() methods
        pred_pos: (T, 6) predicted positions [x1,z1,x2,z2,x3,z3]
        t: (T,) time array
        device: torch device
    
    Returns:
        el_residual_per_step: (T-2,) E-L residual at each interior timestep
        el_residual_avg: float, average E-L residual
    """
    model.eval()
    T = len(t)
    
    if T < 3:
        return np.array([float('nan')]), float('nan')
    
    # Extract positions for bodies 1 and 2 (model uses 4 DOF)
    # pred_pos: [x1, z1, x2, z2, x3, z3]
    q_np = pred_pos[:, :4].astype(np.float32)  # [x1, z1, x2, z2]
    
    # Compute velocities via finite differences (central difference for interior)
    dt = t[1] - t[0]
    vel_np = np.zeros_like(q_np)
    vel_np[1:-1] = (q_np[2:] - q_np[:-2]) / (2 * dt)
    vel_np[0] = (q_np[1] - q_np[0]) / dt  # forward diff at start
    vel_np[-1] = (q_np[-1] - q_np[-2]) / dt  # backward diff at end
    
    # Compute accelerations via finite differences
    acc_np = np.zeros_like(q_np)
    acc_np[1:-1] = (q_np[2:] - 2*q_np[1:-1] + q_np[:-2]) / (dt**2)
    
    # Use interior points only (where we have valid central differences)
    q_interior = q_np[1:-1]
    vel_interior = vel_np[1:-1]
    acc_interior = acc_np[1:-1]
    
    # Convert to tensors with grad enabled
    q = torch.from_numpy(q_interior).to(device).requires_grad_(True)
    qdot = torch.from_numpy(vel_interior).to(device).requires_grad_(True)
    qddot = torch.from_numpy(acc_interior).to(device)
    
    # Compute E-L residual using model's method
    with torch.enable_grad():
        try:
            residual = model.compute_el_residual(q, qdot, qddot)
            el_residual_per_step = torch.norm(residual, dim=1).detach().cpu().numpy()
            el_residual_avg = float(el_residual_per_step.mean())
        except Exception as e:
            print(f"Warning in E-L computation: {e}")
            el_residual_per_step = np.full(T-2, float('nan'))
            el_residual_avg = float('nan')
    
    return el_residual_per_step, el_residual_avg


def compute_lagrangian_structure_score(model, pred_pos, t, device='cuda', G=1.0):
    """
    Compute how well the learned Lagrangian matches the expected L = T - V structure.
    
    For a gravitational system:
    - Kinetic energy: T = (1/2) * sum(m_i * |v_i|^2)
    - Potential energy: V = -G * sum(m_i * m_j / |r_ij|)
    - Lagrangian: L = T - V
    
    We compute the "true" Lagrangian from positions/velocities and compare
    to the learned Lagrangian value.
    
    Args:
        model: Trained LNN model
        pred_pos: (T, 6) predicted positions
        t: (T,) time array
        device: torch device
        G: gravitational constant
    
    Returns:
        structure_score_per_step: (T,) relative error |L_learned - L_true| / |L_true|
        structure_score_avg: float, average structure score (lower is better)
    """
    model.eval()
    T = len(t)
    
    # Extract positions for bodies 1 and 2
    q_np = pred_pos[:, :4].astype(np.float32)
    
    # Compute velocities via finite differences
    dt = t[1] - t[0]
    vel_np = np.zeros_like(q_np)
    vel_np[1:-1] = (q_np[2:] - q_np[:-2]) / (2 * dt)
    vel_np[0] = (q_np[1] - q_np[0]) / dt
    vel_np[-1] = (q_np[-1] - q_np[-2]) / dt
    
    # Convert to tensors
    q = torch.from_numpy(q_np).to(device)
    qdot = torch.from_numpy(vel_np).to(device)
    
    # Compute learned Lagrangian
    with torch.no_grad():
        L_learned = model.lagrangian(q, qdot).cpu().numpy().flatten()
    
    # Compute "true" Lagrangian from physics
    # Positions for all 3 bodies
    x1, z1 = pred_pos[:, 0], pred_pos[:, 1]
    x2, z2 = pred_pos[:, 2], pred_pos[:, 3]
    x3, z3 = pred_pos[:, 4], pred_pos[:, 5]
    
    # Velocities (via finite difference on full 6D)
    vel_full = np.zeros_like(pred_pos)
    vel_full[1:-1] = (pred_pos[2:] - pred_pos[:-2]) / (2 * dt)
    vel_full[0] = (pred_pos[1] - pred_pos[0]) / dt
    vel_full[-1] = (pred_pos[-1] - pred_pos[-2]) / dt
    
    vx1, vz1 = vel_full[:, 0], vel_full[:, 1]
    vx2, vz2 = vel_full[:, 2], vel_full[:, 3]
    vx3, vz3 = vel_full[:, 4], vel_full[:, 5]
    
    # Kinetic energy (m=1 for all)
    T_true = 0.5 * (vx1**2 + vz1**2 + vx2**2 + vz2**2 + vx3**2 + vz3**2)
    
    # Potential energy
    r12 = np.sqrt((x2 - x1)**2 + (z2 - z1)**2) + 1e-10
    r13 = np.sqrt((x3 - x1)**2 + (z3 - z1)**2) + 1e-10
    r23 = np.sqrt((x3 - x2)**2 + (z3 - z2)**2) + 1e-10
    V_true = -G * (1/r12 + 1/r13 + 1/r23)  # m=1 for all
    
    L_true = T_true - V_true
    
    # Compute relative error
    # Note: L_learned is from the 4-DOF Lagrangian network
    # We compare the correlation rather than absolute values
    # since the network may learn a scaled/shifted version
    
    # Normalize and compute correlation-based score
    L_learned_norm = (L_learned - L_learned.mean()) / (L_learned.std() + 1e-10)
    L_true_norm = (L_true - L_true.mean()) / (L_true.std() + 1e-10)
    
    # Correlation coefficient (higher is better, so 1 - corr is our "error")
    correlation = np.corrcoef(L_learned_norm, L_true_norm)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    structure_score_per_step = np.abs(L_learned_norm - L_true_norm)
    structure_score_avg = float(1.0 - correlation)  # 0 = perfect, 1 = no correlation
    
    return structure_score_per_step, structure_score_avg


def evaluate_trajectory_lnn(model, pred_pos, true_pos, true_vel, t, ic=None, 
                            norm_stats=None, device='cuda'):
    """
    Evaluate a single trajectory with both Tier A and Tier B (LNN-specific) metrics.
    
    Args:
        model: Trained LNN model
        pred_pos: (T, 6) predicted positions
        true_pos: (T, 6) true positions
        true_vel: (T, 6) true velocities
        t: (T,) time array
        ic: (2,) initial conditions (optional, for compatibility)
        norm_stats: Normalization statistics (optional, for compatibility)
        device: torch device
    
    Returns:
        metrics: dict with all metrics
    """
    # Tier A metrics (from DNN)
    metrics = evaluate_trajectory_with_ode(pred_pos, true_pos, true_vel, t)
    
    # Tier B: LNN-specific metrics
    try:
        el_per_step, el_avg = compute_el_residual_from_trajectory(
            model, pred_pos, t, device
        )
        metrics['el_residual'] = el_avg
        metrics['el_residual_per_step'] = el_per_step
    except Exception as e:
        print(f"Warning: Could not compute E-L residual: {e}")
        metrics['el_residual'] = float('nan')
        metrics['el_residual_per_step'] = np.full(len(t)-2, float('nan'))
    
    try:
        struct_per_step, struct_avg = compute_lagrangian_structure_score(
            model, pred_pos, t, device
        )
        metrics['lagrangian_structure_score'] = struct_avg
        metrics['lagrangian_structure_per_step'] = struct_per_step
    except Exception as e:
        print(f"Warning: Could not compute Lagrangian structure score: {e}")
        metrics['lagrangian_structure_score'] = float('nan')
        metrics['lagrangian_structure_per_step'] = np.full(len(t), float('nan'))
    
    return metrics


def aggregate_metrics_lnn(all_metrics):
    """
    Aggregate metrics across trajectories, including LNN-specific metrics.
    
    Args:
        all_metrics: list of per-trajectory metric dicts
    
    Returns:
        aggregated: dict with mean, std, min, max for each metric
    """
    # Start with Tier A aggregation
    aggregated = aggregate_metrics(all_metrics)
    
    # Add LNN-specific aggregation
    lnn_metrics = ['el_residual', 'lagrangian_structure_score']
    
    for metric in lnn_metrics:
        if metric in all_metrics[0]:
            values = [m[metric] for m in all_metrics if not np.isnan(m.get(metric, float('nan')))]
            if values:
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
    
    return aggregated


def print_metrics_lnn(aggregated, model_name="LNN"):
    """
    Print metrics including LNN-specific ones.
    
    Args:
        aggregated: dict of aggregated metrics
        model_name: name for display
    """
    # Print Tier A metrics
    print_metrics(aggregated, model_name)
    
    # Print LNN-specific metrics (Tier B)
    print(f"\n{'='*50}")
    print(f"Tier B: {model_name}-Specific Metrics")
    print(f"{'='*50}")
    
    if 'el_residual' in aggregated:
        m = aggregated['el_residual']
        print(f"Euler-Lagrange Residual: {m['mean']:.6f} ± {m['std']:.6f}")
        print(f"  (measures if trajectory satisfies E-L equations; lower is better)")
    
    if 'lagrangian_structure_score' in aggregated:
        m = aggregated['lagrangian_structure_score']
        print(f"Lagrangian Structure Score: {m['mean']:.6f} ± {m['std']:.6f}")
        print(f"  (measures L_learned vs L_true correlation; 0=perfect, 1=no correlation)")


if __name__ == "__main__":
    # Test that imports work
    print("Testing LNN metrics (Tier A from DNN + Tier B LNN-specific)...")
    
    T = 50
    np.random.seed(42)
    
    true_pos = np.random.randn(T, 6).astype(np.float32)
    pred_pos = true_pos + 0.1 * np.random.randn(T, 6).astype(np.float32)
    true_vel = np.random.randn(T, 6).astype(np.float32) * 0.5
    
    print(f"MAE: {mae_state(pred_pos, true_pos):.6f}")
    print(f"RMSE: {rmse_state(pred_pos, true_pos):.6f}")
    print(f"Rollout Error: {rollout_error_avg(pred_pos, true_pos):.6f}")
    print(f"Energy Drift: {energy_drift(pred_pos, true_vel):.6f}")
    
    print("\nLNN metrics module loaded successfully.")
    print("Tier B metrics (E-L Residual, Lagrangian Structure) require model for computation.")
