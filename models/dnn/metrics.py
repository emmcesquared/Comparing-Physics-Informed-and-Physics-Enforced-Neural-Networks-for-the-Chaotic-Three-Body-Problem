"""
Tier A Evaluation Metrics for Three-Body Problem.

All metrics from metrics_reference.md implemented here.
These apply to ALL model architectures (DNN, PINN, LNN, HNN).
"""

import numpy as np
import torch


def compute_energy(positions, velocities, G=1.0, masses=None):
    """
    Compute total mechanical energy for three-body system.

    Args:
        positions: (N, 6) array [x1,z1,x2,z2,x3,z3] or (6,) for single state
        velocities: (N, 6) array [vx1,vz1,vx2,vz2,vx3,vz3] or (6,) for single state
        G: gravitational constant
        masses: list of 3 masses, default [1,1,1]

    Returns:
        energy: scalar or (N,) array
    """
    if masses is None:
        masses = np.array([1.0, 1.0, 1.0])

    single = positions.ndim == 1
    if single:
        positions = positions[np.newaxis, :]
        velocities = velocities[np.newaxis, :]

    N = len(positions)
    energies = np.zeros(N)

    for i in range(N):
        pos = positions[i]
        vel = velocities[i]

        # Reshape to (3, 2) for x,z coordinates
        p = np.array([[pos[0], pos[1]],   # body 1
                      [pos[2], pos[3]],   # body 2
                      [pos[4], pos[5]]])  # body 3

        v = np.array([[vel[0], vel[1]],
                      [vel[2], vel[3]],
                      [vel[4], vel[5]]])

        # Kinetic energy: KE = 0.5 * sum(m * |v|^2)
        KE = 0.5 * np.sum(masses[:, np.newaxis] * v**2)

        # Potential energy: PE = -G * sum_{i<j} m_i * m_j / |r_ij|
        PE = 0.0
        for j in range(3):
            for k in range(j + 1, 3):
                r_jk = np.linalg.norm(p[j] - p[k])
                PE -= G * masses[j] * masses[k] / max(r_jk, 1e-10)

        energies[i] = KE + PE

    return energies[0] if single else energies


# =============================================================================
# Tier A Metrics
# =============================================================================

def mae_state(pred_positions, true_positions):
    """
    Mean Absolute Error on state (positions).

    MAE_s = (1/T) * sum_t |pred_t - true_t|_1

    Args:
        pred_positions: (T, D) predicted positions
        true_positions: (T, D) ground truth positions

    Returns:
        mae: scalar
    """
    return np.mean(np.abs(pred_positions - true_positions))


def rmse_state(pred_positions, true_positions):
    """
    Root Mean Square Error on state (positions).

    RMSE_s = sqrt((1/T) * sum_t |pred_t - true_t|_2^2)

    Args:
        pred_positions: (T, D) predicted positions
        true_positions: (T, D) ground truth positions

    Returns:
        rmse: scalar
    """
    squared_errors = np.sum((pred_positions - true_positions)**2, axis=1)
    return np.sqrt(np.mean(squared_errors))


def relative_error_per_step(pred_positions, true_positions):
    """
    Per-step relative error.

    RelErr_s(t) = |pred_t - true_t|_2 / |true_t|_2

    Args:
        pred_positions: (T, D) predicted positions
        true_positions: (T, D) ground truth positions

    Returns:
        rel_errors: (T,) array of relative errors per time step
    """
    error_norms = np.linalg.norm(pred_positions - true_positions, axis=1)
    true_norms = np.linalg.norm(true_positions, axis=1)
    # Avoid division by zero
    true_norms = np.maximum(true_norms, 1e-10)
    return error_norms / true_norms


def rollout_error_avg(pred_positions, true_positions):
    """
    Time-averaged rollout error.

    RolloutErrAvg = (1/T) * sum_t (|pred_t - true_t|_2 / |true_t|_2)

    Args:
        pred_positions: (T, D) predicted positions
        true_positions: (T, D) ground truth positions

    Returns:
        avg_error: scalar
    """
    rel_errors = relative_error_per_step(pred_positions, true_positions)
    return np.mean(rel_errors)


def absolute_energy_error(pred_positions, true_positions, true_velocities, G=1.0):
    """
    Absolute energy error at each time step.

    Note: For DNN, we don't predict velocities, so we compute energy
    from predicted positions assuming the model captures dynamics.
    This uses true velocities as a proxy.

    AbsEnergyErr(t) = |E_pred_t - E_true_t|

    Args:
        pred_positions: (T, 6) predicted [x1,z1,x2,z2,x3,z3]
        true_positions: (T, 6) ground truth positions
        true_velocities: (T, 6) ground truth velocities

    Returns:
        energy_errors: (T,) array
    """
    # True energy
    E_true = compute_energy(true_positions, true_velocities, G)

    # For predicted, we use true velocities (limitation of position-only models)
    E_pred = compute_energy(pred_positions, true_velocities, G)

    return np.abs(E_pred - E_true)


def energy_drift(pred_positions, true_velocities, G=1.0):
    """
    Maximum energy drift from initial energy.

    EnergyDrift = max_t |E_pred_t - E_pred_0|

    Args:
        pred_positions: (T, 6) predicted positions
        true_velocities: (T, 6) velocities (used for energy computation)

    Returns:
        drift: scalar
    """
    E_pred = compute_energy(pred_positions, true_velocities, G)
    return np.max(np.abs(E_pred - E_pred[0]))


def long_horizon_error(pred_positions, true_positions, horizon_idx=-1):
    """
    Relative error at a specific horizon (default: final time).

    RolloutErr(T*) = |pred_{T*} - true_{T*}|_2 / |true_{T*}|_2

    Args:
        pred_positions: (T, D) predicted positions
        true_positions: (T, D) ground truth positions
        horizon_idx: index of horizon to evaluate (default -1 = final)

    Returns:
        error: scalar
    """
    pred = pred_positions[horizon_idx]
    true = true_positions[horizon_idx]
    return np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-10)


# =============================================================================
# Aggregate Metrics for Multiple Trajectories
# =============================================================================

def evaluate_trajectory(pred_positions, true_positions, true_velocities, t=None):
    """
    Compute all Tier A metrics for a single trajectory.

    Args:
        pred_positions: (T, 6) predicted [x1,z1,x2,z2,x3,z3]
        true_positions: (T, 6) ground truth
        true_velocities: (T, 6) ground truth velocities
        t: (T,) time array (optional, for plotting)

    Returns:
        dict of metrics
    """
    metrics = {
        'mae': mae_state(pred_positions, true_positions),
        'rmse': rmse_state(pred_positions, true_positions),
        'rollout_err_avg': rollout_error_avg(pred_positions, true_positions),
        'long_horizon_err': long_horizon_error(pred_positions, true_positions),
        'energy_drift': energy_drift(pred_positions, true_velocities),
    }

    # Per-step metrics for plotting
    metrics['rel_err_per_step'] = relative_error_per_step(pred_positions, true_positions)
    metrics['energy_err_per_step'] = absolute_energy_error(
        pred_positions, true_positions, true_velocities
    )

    return metrics


# =============================================================================
# ODE Residual (Post-hoc Physics Compliance)
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

    for t_idx in range(T):
        pos = positions[t_idx]
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
            acc[t_idx, i*2:(i+1)*2] = a_i

    return acc


def numerical_acceleration(positions, t):
    """
    Compute acceleration via numerical differentiation of positions.

    Uses second-order central differences.

    Args:
        positions: (T, D) position trajectory
        t: (T,) time array

    Returns:
        accelerations: (T, D) numerical accelerations
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
    Compute ODE residual (physics compliance metric).

    R_ODE(t) = |q̈_t^(pred) - F(q_t)|₂
    ODERes = (1/T) * Σ R_ODE(t)

    This measures how well the predicted trajectory satisfies Newton's laws.

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


def evaluate_trajectory_with_ode(pred_positions, true_positions, true_velocities, t=None):
    """
    Compute all Tier A metrics plus ODE residual for a single trajectory.

    Args:
        pred_positions: (T, 6) predicted [x1,z1,x2,z2,x3,z3]
        true_positions: (T, 6) ground truth
        true_velocities: (T, 6) ground truth velocities
        t: (T,) time array (required for ODE residual)

    Returns:
        dict of metrics including ODE residual
    """
    metrics = {
        'mae': mae_state(pred_positions, true_positions),
        'rmse': rmse_state(pred_positions, true_positions),
        'rollout_err_avg': rollout_error_avg(pred_positions, true_positions),
        'long_horizon_err': long_horizon_error(pred_positions, true_positions),
        'energy_drift': energy_drift(pred_positions, true_velocities),
    }

    # Per-step metrics for plotting
    metrics['rel_err_per_step'] = relative_error_per_step(pred_positions, true_positions)
    metrics['energy_err_per_step'] = absolute_energy_error(
        pred_positions, true_positions, true_velocities
    )

    # ODE residual
    if t is not None:
        ode_res_mean, ode_res_per_step = ode_residual(pred_positions, t)
        metrics['ode_residual'] = ode_res_mean
        metrics['ode_res_per_step'] = ode_res_per_step

    return metrics


def aggregate_metrics(all_metrics):
    """
    Aggregate metrics across multiple trajectories.

    Args:
        all_metrics: list of metric dicts from evaluate_trajectory

    Returns:
        dict with mean and std of each scalar metric
    """
    scalar_keys = ['mae', 'rmse', 'rollout_err_avg', 'long_horizon_err', 'energy_drift']

    aggregated = {}
    for key in scalar_keys:
        values = [m[key] for m in all_metrics]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
        aggregated[f'{key}_median'] = np.median(values)

    # Include ODE residual if present
    if 'ode_residual' in all_metrics[0]:
        ode_values = [m['ode_residual'] for m in all_metrics]
        aggregated['ode_residual_mean'] = np.mean(ode_values)
        aggregated['ode_residual_std'] = np.std(ode_values)
        aggregated['ode_residual_median'] = np.median(ode_values)

    return aggregated


def print_metrics(aggregated, model_name="Model"):
    """Pretty print aggregated metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print('='*60)

    print("\n--- Tier A Metrics ---")
    print(f"MAE (state):           {aggregated['mae_mean']:.6f} +/- {aggregated['mae_std']:.6f}")
    print(f"RMSE (state):          {aggregated['rmse_mean']:.6f} +/- {aggregated['rmse_std']:.6f}")
    print(f"Rollout Error (avg):   {aggregated['rollout_err_avg_mean']:.6f} +/- {aggregated['rollout_err_avg_std']:.6f}")
    print(f"Long-Horizon Error:    {aggregated['long_horizon_err_mean']:.6f} +/- {aggregated['long_horizon_err_std']:.6f}")
    print(f"Energy Drift:          {aggregated['energy_drift_mean']:.6f} +/- {aggregated['energy_drift_std']:.6f}")

    # Print ODE residual if present
    if 'ode_residual_mean' in aggregated:
        print("\n--- Physics Compliance (Post-hoc) ---")
        print(f"ODE Residual:          {aggregated['ode_residual_mean']:.6f} +/- {aggregated['ode_residual_std']:.6f}")


if __name__ == "__main__":
    # Test metrics with synthetic data
    T = 100
    D = 6

    np.random.seed(42)
    true_pos = np.random.randn(T, D)
    pred_pos = true_pos + 0.1 * np.random.randn(T, D)  # Add noise
    true_vel = np.random.randn(T, D) * 0.5

    print("Testing metrics with synthetic data...")
    metrics = evaluate_trajectory(pred_pos, true_pos, true_vel)

    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Rollout Error Avg: {metrics['rollout_err_avg']:.6f}")
    print(f"Long-Horizon Error: {metrics['long_horizon_err']:.6f}")
    print(f"Energy Drift: {metrics['energy_drift']:.6f}")
