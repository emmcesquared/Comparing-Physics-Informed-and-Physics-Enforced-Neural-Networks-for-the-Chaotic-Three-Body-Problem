"""
Evaluation Metrics for HNN on Three-Body Problem.

Tier A (Common to all models - imported from DNN):
- MAE / RMSE on state
- Relative rollout error
- Energy drift
- Long-horizon error
- ODE residual

Tier B (HNN-Specific):
- Hamiltonian Vector-Field Residual: R_H(t) = ||J * grad(H)(s_t) - s_dot_t||_2
  measures if the learned Hamiltonian generates the correct equations of motion
- Hamiltonian MSE: (1/T) * sum (H_learned - H_true)^2
  measures if the learned Hamiltonian matches the true total energy
"""

import importlib.util
from pathlib import Path
import numpy as np
import torch

# Import all metrics from DNN using explicit file loading
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
# Tier B: HNN-Specific Metrics
# =============================================================================

def compute_hamiltonian_vector_field_residual(model, pred_pos, t, device='cuda'):
    """
    Compute Hamiltonian vector-field residual for a predicted trajectory.

    From the metrics spec:
        R_H(t) = ||J * grad(H)(s_t) - s_dot_t||_2

    Where:
    - s = [q, p] is the phase-space state
    - J = [[0, I], [-I, 0]] is the symplectic matrix
    - J * grad(H) = [dH/dp, -dH/dq] (Hamilton's equations)
    - s_dot = [dq/dt, dp/dt] (actual time derivatives)

    For m=1: p = v, so we use velocities as momenta.

    Args:
        model: Trained HNN model with hamiltonian and
               compute_hamiltonian_vector_field_residual methods
        pred_pos: (T, 6) predicted positions [x1,z1,x2,z2,x3,z3]
        t: (T,) time array
        device: torch device

    Returns:
        h_residual_per_step: (T-2,) Hamiltonian residual at each interior timestep
        h_residual_avg: float, average residual (HRes from metrics spec)
    """
    model.eval()
    T = len(t)

    if T < 3:
        return np.array([float('nan')]), float('nan')

    # Extract positions for bodies 1 and 2
    q_np = pred_pos[:, :4].astype(np.float32)  # [x1, z1, x2, z2]

    # Compute velocities via central finite differences
    dt = t[1] - t[0]
    vel_np = np.zeros_like(q_np)
    vel_np[1:-1] = (q_np[2:] - q_np[:-2]) / (2 * dt)
    vel_np[0] = (q_np[1] - q_np[0]) / dt
    vel_np[-1] = (q_np[-1] - q_np[-2]) / dt

    # Compute accelerations (dp/dt for m=1)
    acc_np = np.zeros_like(q_np)
    acc_np[1:-1] = (q_np[2:] - 2*q_np[1:-1] + q_np[:-2]) / (dt**2)

    # Use interior points only
    q_interior = q_np[1:-1]
    vel_interior = vel_np[1:-1]   # = p (momenta, since m=1)
    acc_interior = acc_np[1:-1]   # = dp/dt

    # Convert to tensors with grad enabled
    q = torch.from_numpy(q_interior).to(device).requires_grad_(True)
    p = torch.from_numpy(vel_interior).to(device).requires_grad_(True)
    qdot_actual = torch.from_numpy(vel_interior).to(device)
    pdot_actual = torch.from_numpy(acc_interior).to(device)

    # Compute Hamiltonian vector field
    with torch.enable_grad():
        try:
            dq_dt_H, dp_dt_H = model.compute_hamiltonian_vector_field_residual(q, p)

            # R_H(t) = ||J*grad(H) - s_dot||_2
            # s_dot = [dq/dt, dp/dt], J*grad(H) = [dH/dp, -dH/dq]
            residual_q = dq_dt_H - qdot_actual   # dH/dp - dq/dt
            residual_p = dp_dt_H - pdot_actual    # -dH/dq - dp/dt

            # Full residual: concatenate and take norm
            full_residual = torch.cat([residual_q, residual_p], dim=1)  # (T-2, 8)
            h_residual_per_step = torch.norm(full_residual, dim=1).detach().cpu().numpy()
            h_residual_avg = float(h_residual_per_step.mean())
        except Exception as e:
            print(f"Warning in Hamiltonian residual computation: {e}")
            h_residual_per_step = np.full(T-2, float('nan'))
            h_residual_avg = float('nan')

    return h_residual_per_step, h_residual_avg


def compute_hamiltonian_mse(model, pred_pos, t, device='cuda', G=1.0):
    """
    Compute Hamiltonian MSE: (1/T) * sum (H_learned - H_true)^2

    From the metrics spec:
        MSE_H = (1/T) * sum_t (H_hat_t - H_t)^2

    This measures whether the learned Hamiltonian matches the true total energy.

    Args:
        model: Trained HNN model
        pred_pos: (T, 6) predicted positions
        t: (T,) time array
        device: torch device
        G: gravitational constant

    Returns:
        h_mse_per_step: (T,) per-step squared error
        h_mse_avg: float, average Hamiltonian MSE
    """
    model.eval()
    T = len(t)

    # Extract positions
    q_np = pred_pos[:, :4].astype(np.float32)

    # Compute velocities via finite differences
    dt = t[1] - t[0]
    vel_np = np.zeros_like(q_np)
    vel_np[1:-1] = (q_np[2:] - q_np[:-2]) / (2 * dt)
    vel_np[0] = (q_np[1] - q_np[0]) / dt
    vel_np[-1] = (q_np[-1] - q_np[-2]) / dt

    # Convert to tensors
    q = torch.from_numpy(q_np).to(device)
    p = torch.from_numpy(vel_np).to(device)  # p = v (m=1)

    # Compute learned Hamiltonian
    with torch.no_grad():
        H_learned = model.hamiltonian(q, p).cpu().numpy().flatten()

    # Compute true Hamiltonian H = T + V
    x1, z1 = pred_pos[:, 0], pred_pos[:, 1]
    x2, z2 = pred_pos[:, 2], pred_pos[:, 3]
    x3, z3 = pred_pos[:, 4], pred_pos[:, 5]

    # Full velocities (all 3 bodies)
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
    V_true = -G * (1/r12 + 1/r13 + 1/r23)

    H_true = T_true + V_true

    # MSE per step
    h_mse_per_step = (H_learned - H_true) ** 2
    h_mse_avg = float(h_mse_per_step.mean())

    return h_mse_per_step, h_mse_avg


def evaluate_trajectory_hnn(model, pred_pos, true_pos, true_vel, t, ic=None,
                            norm_stats=None, device='cuda'):
    """
    Evaluate a single trajectory with both Tier A and Tier B (HNN-specific) metrics.

    Args:
        model: Trained HNN model
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

    # Tier B: HNN-specific metrics
    try:
        h_res_per_step, h_res_avg = compute_hamiltonian_vector_field_residual(
            model, pred_pos, t, device
        )
        metrics['hamiltonian_residual'] = h_res_avg
        metrics['hamiltonian_residual_per_step'] = h_res_per_step
    except Exception as e:
        print(f"Warning: Could not compute Hamiltonian residual: {e}")
        metrics['hamiltonian_residual'] = float('nan')
        metrics['hamiltonian_residual_per_step'] = np.full(len(t)-2, float('nan'))

    try:
        h_mse_per_step, h_mse_avg = compute_hamiltonian_mse(
            model, pred_pos, t, device
        )
        metrics['hamiltonian_mse'] = h_mse_avg
        metrics['hamiltonian_mse_per_step'] = h_mse_per_step
    except Exception as e:
        print(f"Warning: Could not compute Hamiltonian MSE: {e}")
        metrics['hamiltonian_mse'] = float('nan')
        metrics['hamiltonian_mse_per_step'] = np.full(len(t), float('nan'))

    return metrics


def aggregate_metrics_hnn(all_metrics):
    """
    Aggregate metrics across trajectories, including HNN-specific metrics.

    Args:
        all_metrics: list of per-trajectory metric dicts

    Returns:
        aggregated: dict with mean, std, min, max for each metric
    """
    # Start with Tier A aggregation
    aggregated = aggregate_metrics(all_metrics)

    # Add HNN-specific aggregation
    hnn_metrics = ['hamiltonian_residual', 'hamiltonian_mse']

    for metric in hnn_metrics:
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


def print_metrics_hnn(aggregated, model_name="HNN"):
    """
    Print metrics including HNN-specific ones.

    Args:
        aggregated: dict of aggregated metrics
        model_name: name for display
    """
    # Print Tier A metrics
    print_metrics(aggregated, model_name)

    # Print HNN-specific metrics (Tier B)
    print(f"\n{'='*50}")
    print(f"Tier B: {model_name}-Specific Metrics")
    print(f"{'='*50}")

    if 'hamiltonian_residual' in aggregated:
        m = aggregated['hamiltonian_residual']
        print(f"Hamiltonian Vector-Field Residual: {m['mean']:.6f} +/- {m['std']:.6f}")
        print(f"  (R_H = ||J*grad(H) - s_dot||; measures if H generates correct motion; lower is better)")

    if 'hamiltonian_mse' in aggregated:
        m = aggregated['hamiltonian_mse']
        print(f"Hamiltonian MSE: {m['mean']:.6f} +/- {m['std']:.6f}")
        print(f"  (MSE between learned H and true H = T + V; lower is better)")


if __name__ == "__main__":
    # Test that imports work
    print("Testing HNN metrics (Tier A from DNN + Tier B HNN-specific)...")

    T = 50
    np.random.seed(42)

    true_pos = np.random.randn(T, 6).astype(np.float32)
    pred_pos = true_pos + 0.1 * np.random.randn(T, 6).astype(np.float32)
    true_vel = np.random.randn(T, 6).astype(np.float32) * 0.5

    print(f"MAE: {mae_state(pred_pos, true_pos):.6f}")
    print(f"RMSE: {rmse_state(pred_pos, true_pos):.6f}")
    print(f"Rollout Error: {rollout_error_avg(pred_pos, true_pos):.6f}")
    print(f"Energy Drift: {energy_drift(pred_pos, true_vel):.6f}")

    print("\nHNN metrics module loaded successfully.")
    print("Tier B metrics (Hamiltonian Residual, Hamiltonian MSE) require model for computation.")
