"""
Comprehensive analysis of the PINN Three-Body Dataset.
Generates validation statistics and visualization plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_dataset(filepath="pinn_three_body_dataset.npz"):
    """Load the dataset and return arrays."""
    data = np.load(filepath)
    t = data['t']
    Y = data['Y']
    y0 = data['y0']
    failed_indices = data['failed_indices']
    return t, Y, y0, failed_indices


def compute_energy(state, G=1.0, masses=None):
    """
    Compute total energy (kinetic + potential) for a single state.

    Args:
        state: 18-dimensional state vector
        G: gravitational constant
        masses: array of 3 masses

    Returns:
        total_energy: scalar
    """
    if masses is None:
        masses = np.array([1.0, 1.0, 1.0])

    pos = state[:9].reshape(3, 3)
    vel = state[9:18].reshape(3, 3)

    # Kinetic energy
    KE = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)

    # Potential energy
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r_ij = np.linalg.norm(pos[i] - pos[j])
            PE -= G * masses[i] * masses[j] / max(r_ij, 1e-10)

    return KE + PE


def compute_trajectory_energies(trajectory):
    """Compute energy at each time step for a trajectory."""
    return np.array([compute_energy(state) for state in trajectory])


def compute_minimum_separation(trajectory):
    """Compute minimum pairwise separation across all time steps."""
    min_sep = np.inf
    for state in trajectory:
        pos = state[:9].reshape(3, 3)
        for i in range(3):
            for j in range(i + 1, 3):
                sep = np.linalg.norm(pos[i] - pos[j])
                min_sep = min(min_sep, sep)
    return min_sep


def analyze_dataset(t, Y, y0):
    """Perform comprehensive analysis of the dataset."""
    n_traj = len(Y)
    n_time = len(t)

    print("=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)

    # Basic info
    print("\n--- Dataset Overview ---")
    print(f"Trajectories: {n_traj}")
    print(f"Time points: {n_time} (t in [{t[0]}, {t[-1]}])")
    print(f"Time step (dt): {t[1] - t[0]:.6f}")
    print(f"State dimension: 18 (9 positions + 9 velocities)")

    # Validate initial conditions
    print("\n--- Initial Condition Validation ---")

    # Check particle 1 at (1, 0, 0)
    p1_positions = y0[:, 0:3]
    p1_deviation = np.max(np.abs(p1_positions - [1.0, 0.0, 0.0]))
    print(f"Particle 1 at (1,0,0): max deviation = {p1_deviation:.2e}")

    # Check all initial velocities = 0
    init_velocities = y0[:, 9:18]
    vel_deviation = np.max(np.abs(init_velocities))
    print(f"Initial velocities = 0: max deviation = {vel_deviation:.2e}")

    # Check center of mass
    com_deviations = []
    for i in range(n_traj):
        positions = y0[i, 0:9].reshape(3, 3)
        com = positions.mean(axis=0)
        com_deviations.append(np.linalg.norm(com))
    print(f"Center of mass at origin: max deviation = {max(com_deviations):.2e}")

    # Particle 2 range
    p2_x = y0[:, 3]
    p2_z = y0[:, 5]
    print(f"Particle 2 X range: [{p2_x.min():.3f}, {p2_x.max():.3f}] (expected [0, 0.5])")
    print(f"Particle 2 Z range: [{p2_z.min():.3f}, {p2_z.max():.3f}] (expected [0, 1])")

    # Particle 3 range
    p3_x = y0[:, 6]
    p3_z = y0[:, 8]
    print(f"Particle 3 X range: [{p3_x.min():.3f}, {p3_x.max():.3f}] (expected [-1.5, -1])")
    print(f"Particle 3 Z range: [{p3_z.min():.3f}, {p3_z.max():.3f}] (expected [-1, 0])")

    # Energy conservation
    print("\n--- Energy Conservation ---")
    initial_energies = []
    energy_errors = []

    for i in range(n_traj):
        energies = compute_trajectory_energies(Y[i])
        E0 = energies[0]
        initial_energies.append(E0)
        rel_error = np.max(np.abs(energies - E0)) / np.abs(E0)
        energy_errors.append(rel_error)

    initial_energies = np.array(initial_energies)
    energy_errors = np.array(energy_errors)

    print(f"Initial energy range: [{initial_energies.min():.2f}, {initial_energies.max():.2f}]")
    print(f"All bound systems (E < 0): {np.all(initial_energies < 0)}")
    print(f"Mean relative energy error: {energy_errors.mean()*100:.4f}%")
    print(f"Max relative energy error: {energy_errors.max()*100:.4f}%")
    print(f"Median relative energy error: {np.median(energy_errors)*100:.6f}%")

    # Chaos and dynamics
    print("\n--- Chaos & Dynamics ---")

    # Position spread
    positions_t0 = Y[:, 0, :9].reshape(n_traj, 3, 3)
    positions_t_end = Y[:, -1, :9].reshape(n_traj, 3, 3)

    spread_t0 = np.std(positions_t0)
    spread_t_end = np.std(positions_t_end)

    print(f"Position spread at t=0: {spread_t0:.2f}")
    print(f"Position spread at t={t[-1]}: {spread_t_end:.2f} ({spread_t_end/spread_t0:.1f}× increase)")

    # Velocity statistics
    all_velocities = Y[:, :, 9:18]
    max_vel = np.max(np.linalg.norm(all_velocities.reshape(-1, 3, 3), axis=2))

    max_vels_per_traj = []
    for i in range(n_traj):
        vels = Y[i, :, 9:18].reshape(-1, 3, 3)
        max_v = np.max(np.linalg.norm(vels, axis=2))
        max_vels_per_traj.append(max_v)
    max_vels_per_traj = np.array(max_vels_per_traj)

    print(f"Max velocity reached: {max_vel:.2f}")
    print(f"Mean max velocity: {max_vels_per_traj.mean():.2f} ± {max_vels_per_traj.std():.2f}")

    # Maximum displacement
    max_displacements = []
    for i in range(n_traj):
        pos = Y[i, :, :9].reshape(-1, 3, 3)
        displacement = np.max(np.linalg.norm(pos, axis=2))
        max_displacements.append(displacement)
    max_displacements = np.array(max_displacements)
    print(f"Max displacement from origin: {max_displacements.max():.1f}")

    # Close encounters
    print("\n--- Close Encounters ---")
    min_separations = []
    for i in range(n_traj):
        min_sep = compute_minimum_separation(Y[i])
        min_separations.append(min_sep)
    min_separations = np.array(min_separations)

    close_threshold = 0.1
    close_count = np.sum(min_separations < close_threshold)
    print(f"Trajectories with close encounters (< {close_threshold}): {close_count}/{n_traj} ({100*close_count/n_traj:.0f}%)")
    print(f"Minimum separation observed: {min_separations.min():.4f}")
    print(f"Mean minimum separation: {min_separations.mean():.3f}")

    return {
        'initial_energies': initial_energies,
        'energy_errors': energy_errors,
        'min_separations': min_separations,
        'max_vels_per_traj': max_vels_per_traj,
        'max_displacements': max_displacements
    }


def plot_sample_trajectories(t, Y, save_path="sample_trajectories.png"):
    """Plot 6 sample trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = ['#e74c3c', '#3498db', '#2ecc71']
    labels = ['Body 1', 'Body 2', 'Body 3']

    np.random.seed(42)
    indices = np.random.choice(len(Y), 6, replace=False)

    for idx, (ax, traj_idx) in enumerate(zip(axes, indices)):
        trajectory = Y[traj_idx]

        for body in range(3):
            x = trajectory[:, body * 3]
            z = trajectory[:, body * 3 + 2]

            ax.plot(x, z, color=colors[body], label=labels[body], alpha=0.8, linewidth=1)
            ax.scatter(x[0], z[0], color=colors[body], marker='o', s=80, edgecolors='black', zorder=5)
            ax.scatter(x[-1], z[-1], color=colors[body], marker='s', s=60, edgecolors='black', zorder=5)

        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Z', fontsize=10)
        ax.set_title(f'Trajectory {traj_idx}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

    plt.suptitle('Sample Three-Body Trajectories (x-z plane)\nCircles: start, Squares: end', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_initial_condition_distribution(y0, save_path="initial_condition_distribution.png"):
    """Plot the initial condition distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract positions
    p1 = y0[:, 0:3]
    p2 = y0[:, 3:6]
    p3 = y0[:, 6:9]

    # Scatter plot of initial positions (x-z view)
    ax = axes[0]
    ax.scatter(p1[:, 0], p1[:, 2], alpha=0.5, label='Body 1', s=20, c='#e74c3c')
    ax.scatter(p2[:, 0], p2[:, 2], alpha=0.5, label='Body 2', s=20, c='#3498db')
    ax.scatter(p3[:, 0], p3[:, 2], alpha=0.5, label='Body 3', s=20, c='#2ecc71')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Initial Positions (x-z plane)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Histogram of p2 positions
    ax = axes[1]
    ax.hist(p2[:, 0], bins=30, alpha=0.7, label='X', color='#3498db')
    ax.hist(p2[:, 2], bins=30, alpha=0.7, label='Z', color='#2ecc71')
    ax.set_xlabel('Position')
    ax.set_ylabel('Count')
    ax.set_title('Body 2 Position Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Semi-disk visualization for p2
    ax = axes[2]
    ax.scatter(p2[:, 0], p2[:, 2], alpha=0.5, s=10, c='#3498db')

    # Draw expected semi-disk boundary
    theta = np.linspace(0, np.pi/2, 100)
    x_boundary = np.minimum(0.5, np.cos(theta))
    z_boundary = np.sin(theta)
    ax.plot(x_boundary, z_boundary, 'r-', linewidth=2, label='Semi-disk boundary')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Body 2 Semi-Disk Sampling (Algorithm 1)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 0.6)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_energy_conservation(t, Y, save_path="energy_conservation.png"):
    """Plot energy conservation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute energies for all trajectories
    n_traj = len(Y)
    all_energies = []
    rel_errors = []

    sample_indices = np.random.choice(n_traj, min(20, n_traj), replace=False)

    for i in range(n_traj):
        energies = compute_trajectory_energies(Y[i])
        all_energies.append(energies)
        E0 = energies[0]
        rel_err = (energies - E0) / np.abs(E0)
        rel_errors.append(rel_err)

    all_energies = np.array(all_energies)
    rel_errors = np.array(rel_errors)

    # Plot 1: Sample energy vs time
    ax = axes[0, 0]
    for i in sample_indices[:10]:
        ax.plot(t, all_energies[i], alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy vs Time (10 sample trajectories)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Relative energy error vs time
    ax = axes[0, 1]
    mean_error = np.mean(rel_errors, axis=0)
    std_error = np.std(rel_errors, axis=0)
    ax.plot(t, mean_error * 100, 'b-', linewidth=2, label='Mean')
    ax.fill_between(t, (mean_error - std_error) * 100, (mean_error + std_error) * 100,
                    alpha=0.3, label='±1 std')
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Energy Error (%)')
    ax.set_title('Mean Relative Energy Error over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Initial energy distribution
    ax = axes[1, 0]
    initial_E = all_energies[:, 0]
    ax.hist(initial_E, bins=40, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(initial_E.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {initial_E.mean():.2f}')
    ax.set_xlabel('Initial Energy')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Initial Energies')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Max relative error distribution
    ax = axes[1, 1]
    max_errors = np.max(np.abs(rel_errors), axis=1) * 100
    ax.hist(max_errors, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.axvline(max_errors.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {max_errors.mean():.4f}%')
    ax.set_xlabel('Max Relative Energy Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Maximum Energy Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_errors.max() * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_chaos_analysis(t, Y, stats, save_path="chaos_analysis.png"):
    """Plot chaos and dynamics analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_traj = len(Y)

    # Plot 1: Position spread evolution
    ax = axes[0, 0]
    spreads = []
    for ti in range(len(t)):
        pos = Y[:, ti, :9].reshape(n_traj, 3, 3)
        spreads.append(np.std(pos))
    ax.plot(t, spreads, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position Spread (std)')
    ax.set_title('Position Spread Evolution')
    ax.grid(True, alpha=0.3)

    # Plot 2: Minimum separation distribution
    ax = axes[0, 1]
    ax.hist(stats['min_separations'], bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Close encounter threshold (0.1)')
    ax.axvline(stats['min_separations'].mean(), color='blue', linestyle='--', linewidth=2,
               label=f"Mean: {stats['min_separations'].mean():.3f}")
    ax.set_xlabel('Minimum Separation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Minimum Separations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Max velocity distribution
    ax = axes[1, 0]
    ax.hist(stats['max_vels_per_traj'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.axvline(stats['max_vels_per_traj'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {stats['max_vels_per_traj'].mean():.2f}")
    ax.set_xlabel('Maximum Velocity')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Maximum Velocities per Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Trajectory evolution example (showing chaos)
    ax = axes[1, 1]
    # Pick 5 trajectories with similar initial conditions
    y0 = Y[:, 0, :]
    base_idx = 0
    distances = np.linalg.norm(y0 - y0[base_idx], axis=1)
    similar_indices = np.argsort(distances)[:5]

    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, idx in enumerate(similar_indices):
        pos = Y[idx, :, :3]  # Body 1 position
        ax.plot(pos[:, 0], pos[:, 2], color=colors[i], alpha=0.8, linewidth=1,
                label=f'Traj {idx}')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Body 1 Trajectories (similar initial conditions)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_distance_evolution(t, Y, save_path="distance_evolution.png"):
    """Plot pairwise distance evolution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pair_names = ['Bodies 1-2', 'Bodies 1-3', 'Bodies 2-3']
    pair_indices = [(0, 1), (0, 2), (1, 2)]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    np.random.seed(42)
    sample_indices = np.random.choice(len(Y), 50, replace=False)

    for ax, (pair_name, (i, j), color) in zip(axes, zip(pair_names, pair_indices, colors)):
        for traj_idx in sample_indices:
            pos = Y[traj_idx, :, :9].reshape(-1, 3, 3)
            dist = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
            ax.plot(t, dist, color=color, alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Distance')
        ax.set_title(f'Pairwise Distance: {pair_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    plt.suptitle('Pairwise Distance Evolution (50 sample trajectories)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Main analysis function."""
    print("Loading dataset...")
    t, Y, y0, failed = load_dataset()

    print(f"Dataset loaded: {len(Y)} trajectories")
    print(f"Failed integrations: {len(failed)}")

    # Run analysis
    stats = analyze_dataset(t, Y, y0)

    # Generate all plots
    print("\n--- Generating Plots ---")
    plot_sample_trajectories(t, Y, "sample_trajectories.png")
    plot_initial_condition_distribution(y0, "initial_condition_distribution.png")
    plot_energy_conservation(t, Y, "energy_conservation.png")
    plot_chaos_analysis(t, Y, stats, "chaos_analysis.png")
    plot_distance_evolution(t, Y, "distance_evolution.png")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  1. sample_trajectories.png - Sample 3-body trajectories")
    print("  2. initial_condition_distribution.png - Initial condition sampling")
    print("  3. energy_conservation.png - Energy conservation analysis")
    print("  4. chaos_analysis.png - Chaos and dynamics metrics")
    print("  5. distance_evolution.png - Pairwise distance evolution")


if __name__ == "__main__":
    main()
