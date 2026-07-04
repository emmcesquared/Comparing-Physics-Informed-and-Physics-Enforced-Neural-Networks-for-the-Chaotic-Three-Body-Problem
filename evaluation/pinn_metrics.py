"""
Evaluation Metrics for PINN Three-Body Problem.

Includes:
1. All Tier A metrics (from DNN metrics)
2. PINN-specific: ODE Residual
"""

import numpy as np
import torch
import os
import sys
import importlib.util

# Load DNN metrics module explicitly to avoid name collision
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dnn_metrics_path = os.path.join(os.path.dirname(_current_dir), 'dnn', 'metrics.py')
_spec = importlib.util.spec_from_file_location("dnn_metrics", _dnn_metrics_path)
_dnn_metrics_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dnn_metrics_module)

# Import Tier A metrics
compute_energy = _dnn_metrics_module.compute_energy
mae_state = _dnn_metrics_module.mae_state
rmse_state = _dnn_metrics_module.rmse_state
relative_error_per_step = _dnn_metrics_module.relative_error_per_step
rollout_error_avg = _dnn_metrics_module.rollout_error_avg
absolute_energy_error = _dnn_metrics_module.absolute_energy_error
energy_drift = _dnn_metrics_module.energy_drift
long_horizon_error = _dnn_metrics_module.long_horizon_error
evaluate_trajectory_base = _dnn_metrics_module.evaluate_trajectory
aggregate_metrics_base = _dnn_metrics_module.aggregate_metrics
print_metrics_base = _dnn_metrics_module.print_metrics


# =============================================================================
# PINN-Specific Metrics (Tier B)
# =============================================================================

def compute_true_acceleration(positions, G=1.0, masses=None):
    """
    Compute true gravitational acceleration for all bodies.

    Args:
        positions: (T, 6) array [x1,z1,x2,z2,x3,z3]
        G: gravitational constant
        masses: array of 3 masses

    Returns:
        accelerations: (T, 6) array [ax1,az1,ax2,az2,ax3,az3]
    """
    if masses is None:
        masses = np.array([1.0, 1.0, 1.0])

    T = len(positions)
    acc = np.zeros((T, 6))

    for t in range(T):
        pos = positions[t]
        # Reshape to (3, 2) for bodies
        p = np.array([
            [pos[0], pos[1]],  # body 1
            [pos[2], pos[3]],  # body 2
            [pos[4], pos[5]]   # body 3
        ])

        # Compute acceleration for each body
        for i in range(3):
            a_i = np.zeros(2)
            for j in range(3):
                if i != j:
                    r_ij = p[j] - p[i]
                    r_norm = np.linalg.norm(r_ij) + 1e-10
                    a_i += G * masses[j] * r_ij / (r_norm ** 3)
            acc[t, i*2:(i+1)*2] = a_i

    return acc


def numerical_acceleration(positions, t):
    """
    Compute acceleration via numerical differentiation of positions.

    Uses second-order central differences.

    Args:
        positions: (T, D) position trajectory
        t: (T,) time array

    Returns:
        accelerations: (T, D) numerical accelerations (edges are forward/backward diff)
    """
    T = len(positions)
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    # First compute velocities
    vel = np.zeros_like(positions)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    for i in range(1, T-1):
        vel[i] = (positions[i+1] - positions[i-1]) / (2 * dt)

    # Then compute accelerations
    acc = np.zeros_like(positions)
    acc[0] = (vel[1] - vel[0]) / dt
    acc[-1] = (vel[-1] - vel[-2]) / dt
    for i in range(1, T-1):
        acc[i] = (vel[i+1] - vel[i-1]) / (2 * dt)

    return acc


def ode_residual(pred_positions, t, G=1.0, masses=None):
    """
    Compute ODE residual (PINN-specific metric).

    R_ODE(t) = |q̈_t^(pred) - F(q_t)|₂
    ODERes = (1/T) * Σ R_ODE(t)

    Args:
        pred_positions: (T, 6) predicted positions [x1,z1,x2,z2,x3,z3]
        t: (T,) time array
        G: gravitational constant
        masses: array of 3 masses

    Returns:
        ode_res_mean: scalar mean residual
        ode_res_per_step: (T,) residual at each time step
    """
    # Numerical acceleration from positions
    pred_acc = numerical_acceleration(pred_positions, t)

    # True acceleration from gravitational law
    true_acc = compute_true_acceleration(pred_positions, G, masses)

    # Residual
    residual = np.linalg.norm(pred_acc - true_acc, axis=1)

    return residual.mean(), residual


# =============================================================================
# Extended Trajectory Evaluation
# =============================================================================

def evaluate_trajectory_pinn(pred_positions, true_positions, true_velocities, t):
    """
    Compute all metrics for a single trajectory (Tier A + PINN-specific).

    Args:
        pred_positions: (T, 6) predicted [x1,z1,x2,z2,x3,z3]
        true_positions: (T, 6) ground truth positions
        true_velocities: (T, 6) ground truth velocities
        t: (T,) time array

    Returns:
        dict of all metrics
    """
    # Get Tier A metrics
    metrics = evaluate_trajectory_base(pred_positions, true_positions, true_velocities, t)

    # Add PINN-specific metrics
    ode_res_mean, ode_res_per_step = ode_residual(pred_positions, t)
    metrics['ode_residual'] = ode_res_mean
    metrics['ode_res_per_step'] = ode_res_per_step

    return metrics


def aggregate_metrics_pinn(all_metrics):
    """
    Aggregate metrics across multiple trajectories (Tier A + PINN-specific).

    Args:
        all_metrics: list of metric dicts

    Returns:
        dict with mean and std of each scalar metric
    """
    # Start with Tier A aggregation
    aggregated = aggregate_metrics_base(all_metrics)

    # Add PINN-specific
    ode_res_values = [m['ode_residual'] for m in all_metrics]
    aggregated['ode_residual_mean'] = np.mean(ode_res_values)
    aggregated['ode_residual_std'] = np.std(ode_res_values)
    aggregated['ode_residual_median'] = np.median(ode_res_values)

    return aggregated


def print_metrics_pinn(aggregated, model_name="PINN"):
    """Pretty print aggregated metrics including PINN-specific."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print('='*60)

    print("\n--- Tier A Metrics ---")
    print(f"MAE (state):           {aggregated['mae_mean']:.6f} +/- {aggregated['mae_std']:.6f}")
    print(f"RMSE (state):          {aggregated['rmse_mean']:.6f} +/- {aggregated['rmse_std']:.6f}")
    print(f"Rollout Error (avg):   {aggregated['rollout_err_avg_mean']:.6f} +/- {aggregated['rollout_err_avg_std']:.6f}")
    print(f"Long-Horizon Error:    {aggregated['long_horizon_err_mean']:.6f} +/- {aggregated['long_horizon_err_std']:.6f}")
    print(f"Energy Drift:          {aggregated['energy_drift_mean']:.6f} +/- {aggregated['energy_drift_std']:.6f}")

    print("\n--- PINN-Specific Metrics (Tier B) ---")
    print(f"ODE Residual:          {aggregated['ode_residual_mean']:.6f} +/- {aggregated['ode_residual_std']:.6f}")


if __name__ == "__main__":
    # Test metrics with synthetic data
    T = 100
    D = 6

    np.random.seed(42)
    true_pos = np.random.randn(T, D)
    pred_pos = true_pos + 0.1 * np.random.randn(T, D)
    true_vel = np.random.randn(T, D) * 0.5
    t = np.linspace(0, 10, T)

    print("Testing PINN metrics with synthetic data...")
    metrics = evaluate_trajectory_pinn(pred_pos, true_pos, true_vel, t)

    print(f"\nTier A:")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Rollout Error Avg: {metrics['rollout_err_avg']:.6f}")
    print(f"  Long-Horizon Error: {metrics['long_horizon_err']:.6f}")
    print(f"  Energy Drift: {metrics['energy_drift']:.6f}")

    print(f"\nPINN-Specific:")
    print(f"  ODE Residual: {metrics['ode_residual']:.6f}")
