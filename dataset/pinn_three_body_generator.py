"""
PINN Paper Three-Body Dataset Generator
Generates planar Newtonian three-body trajectories following the exact specification
from "Physics-Informed Neural Networks for the Three-Body Problem" paper.

Physical Setup:
- G = 1, m1 = m2 = m3 = 1 (dimensionless units)
- Planar motion in x-z plane (y = 0 always)
- 18-dimensional state: 9 positions + 9 velocities

Initial Conditions (Algorithm 1 from paper):
- Particle 1 fixed at (1, 0, 0)
- Particle 2 sampled from semi-disk region
- Particle 3 determined by center-of-mass constraint
- All initial velocities = 0
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


def gravitational_acceleration(t, state, G=1.0, masses=None):
    """
    Compute gravitational accelerations for three bodies.
    
    State vector layout: [x1,y1,z1, x2,y2,z2, x3,y3,z3, vx1,vy1,vz1, vx2,vy2,vz2, vx3,vy3,vz3]
    
    Returns derivatives: [vx1,vy1,vz1, vx2,vy2,vz2, vx3,vy3,vz3, ax1,ay1,az1, ax2,ay2,az2, ax3,ay3,az3]
    """
    if masses is None:
        masses = np.array([1.0, 1.0, 1.0])
    
    # Extract positions
    pos = state[:9].reshape(3, 3)  # 3 bodies x 3 coordinates
    vel = state[9:18]
    
    # Compute accelerations
    acc = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = pos[j] - pos[i]
                r_mag = np.linalg.norm(r_ij)
                # Softening to avoid numerical issues at very close encounters
                r_mag = max(r_mag, 1e-6)
                acc[i] += G * masses[j] * r_ij / (r_mag ** 3)
    
    # Return derivatives: velocities followed by accelerations
    return np.concatenate([vel, acc.flatten()])


def sample_initial_conditions_algorithm1():
    """
    Generate initial conditions following Algorithm 1 from the PINN paper.
    
    Returns:
        initial_state: 18-dimensional state vector
        p1, p2, p3: individual position vectors for verification
    """
    # Particle 1 fixed at (1, 0, 0)
    p1 = np.array([1.0, 0.0, 0.0])
    
    # Particle 2: Semi-disk sampling (Algorithm 1)
    # Step 1: Sample theta uniformly from [0, pi/2]
    theta = np.random.uniform(0, np.pi / 2)
    
    # Step 2: Construct base vector p = (min(0.5, cos(theta)), 0, sin(theta))
    base_x = min(0.5, np.cos(theta))
    base_z = np.sin(theta)
    base_vector = np.array([base_x, 0.0, base_z])
    
    # Step 3: Sample radial scaling s uniformly from [0, 1]
    s = np.random.uniform(0, 1)
    
    # Step 4: Set p2 = s * base_vector
    p2 = s * base_vector
    
    # Particle 3 determined by center-of-mass constraint
    # For equal masses: p1 + p2 + p3 = 0
    p3 = -(p1 + p2)
    
    # All initial velocities are zero
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 0.0])
    v3 = np.array([0.0, 0.0, 0.0])
    
    # Construct 18-dimensional state vector
    positions = np.concatenate([p1, p2, p3])  # 9 components
    velocities = np.concatenate([v1, v2, v3])  # 9 components
    initial_state = np.concatenate([positions, velocities])  # 18 components
    
    return initial_state, p1, p2, p3


def integrate_trajectory(initial_state, t_eval, G=1.0):
    """
    Integrate the three-body system using high-order Runge-Kutta (DOP853).
    
    Args:
        initial_state: 18-dimensional initial state
        t_eval: time points at which to evaluate the solution
        G: gravitational constant
    
    Returns:
        states: trajectory array of shape (len(t_eval), 18)
        success: whether integration completed successfully
    """
    t_span = (t_eval[0], t_eval[-1])
    
    # Use DOP853 (8th order Dormand-Prince) with moderate tolerance for speed
    # max_step prevents the integrator from getting stuck on stiff close-encounters
    try:
        solution = solve_ivp(
            gravitational_acceleration,
            t_span,
            initial_state,
            method='DOP853',
            t_eval=t_eval,
            args=(G,),
            rtol=1e-8,
            atol=1e-10,
            max_step=0.01,  # Prevent getting stuck on stiff sections
            dense_output=False
        )
        
        if solution.success:
            return solution.y.T, True  # Transpose to get (time, state)
        else:
            return None, False
    except Exception as e:
        return None, False


def generate_dataset(num_trajectories=1000, seed=42):
    """
    Generate the full three-body dataset.
    
    Args:
        num_trajectories: number of trajectories to generate
        seed: random seed for reproducibility
    
    Returns:
        t: time array of shape (129,)
        Y: trajectories array of shape (num_trajectories, 129, 18)
        y0: initial conditions array of shape (num_trajectories, 18)
    """
    np.random.seed(seed)
    
    # Time grid: 129 points from 0 to 10.0 (128 intervals)
    t = np.linspace(0.0, 10.0, 129)
    dt = t[1] - t[0]
    
    print(f"Three-Body Dataset Generator (PINN Paper Specification)")
    print(f"=" * 55)
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Time span: [0, 10.0]")
    print(f"Time step (dt): {dt:.6f}")
    print(f"Number of time points: {len(t)}")
    print(f"State dimension: 18 (9 positions + 9 velocities)")
    print(f"Coordinate system: x-z plane (y = 0)")
    print()
    
    # Storage
    Y = np.zeros((num_trajectories, 129, 18))
    y0 = np.zeros((num_trajectories, 18))
    
    successful = 0
    failed_indices = []
    
    print("Generating trajectories...")
    for i in tqdm(range(num_trajectories), desc="Simulating"):
        # Generate initial conditions using Algorithm 1
        initial_state, p1, p2, p3 = sample_initial_conditions_algorithm1()
        
        # Store initial condition
        y0[i] = initial_state
        
        # Integrate trajectory
        trajectory, success = integrate_trajectory(initial_state, t)
        
        if success:
            Y[i] = trajectory
            successful += 1
        else:
            # Keep initial state for failed trajectories (will be flagged)
            Y[i, 0] = initial_state
            failed_indices.append(i)
    
    print()
    print(f"Completed: {successful}/{num_trajectories} successful")
    if failed_indices:
        print(f"Failed indices: {failed_indices}")
    
    return t, Y, y0, failed_indices


def validate_dataset(t, Y, y0):
    """
    Validate the generated dataset meets the specification.
    """
    print("\nValidating dataset...")
    
    errors = []
    
    # Check shapes
    if t.shape != (129,):
        errors.append(f"t shape: expected (129,), got {t.shape}")
    
    if Y.shape[1:] != (129, 18):
        errors.append(f"Y shape: expected (N, 129, 18), got {Y.shape}")
    
    if y0.shape[1] != 18:
        errors.append(f"y0 shape: expected (N, 18), got {y0.shape}")
    
    # Check time grid
    if not np.allclose(t[0], 0.0):
        errors.append(f"t[0]: expected 0.0, got {t[0]}")
    
    if not np.allclose(t[-1], 10.0):
        errors.append(f"t[-1]: expected 10.0, got {t[-1]}")
    
    dt = 10.0 / 128
    if not np.allclose(t[1] - t[0], dt):
        errors.append(f"dt: expected {dt}, got {t[1] - t[0]}")
    
    # Check initial conditions for a sample
    for i in range(min(10, len(y0))):
        # Particle 1 should be at (1, 0, 0)
        p1 = y0[i, 0:3]
        if not np.allclose(p1, [1.0, 0.0, 0.0]):
            errors.append(f"Trajectory {i}: p1 = {p1}, expected [1, 0, 0]")
        
        # All y-coordinates should be 0 (x-z plane)
        y_coords = y0[i, [1, 4, 7]]  # y1, y2, y3
        if not np.allclose(y_coords, 0.0):
            errors.append(f"Trajectory {i}: y-coords = {y_coords}, expected 0")
        
        # All initial velocities should be 0
        velocities = y0[i, 9:18]
        if not np.allclose(velocities, 0.0):
            errors.append(f"Trajectory {i}: initial velocities non-zero")
        
        # Center of mass should be at origin
        positions = y0[i, 0:9].reshape(3, 3)
        com = positions.mean(axis=0)
        if not np.allclose(com, 0.0, atol=1e-10):
            errors.append(f"Trajectory {i}: COM = {com}, expected [0, 0, 0]")
    
    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("Validation PASSED!")
        return True


def visualize_samples(t, Y, num_samples=6, save_path=None):
    """
    Visualize sample trajectories to verify chaotic behavior.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    labels = ['Body 1', 'Body 2', 'Body 3']
    
    indices = np.random.choice(len(Y), min(num_samples, len(Y)), replace=False)
    
    for idx, (ax, traj_idx) in enumerate(zip(axes, indices)):
        trajectory = Y[traj_idx]
        
        # Extract positions (x-z plane view, but stored as x,y,z with y=0)
        for body in range(3):
            x = trajectory[:, body * 3]      # x coordinate
            z = trajectory[:, body * 3 + 2]  # z coordinate
            
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
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to generate and save the dataset."""
    
    # Generate dataset
    t, Y, y0, failed = generate_dataset(num_trajectories=1000, seed=42)
    
    # Validate
    validate_dataset(t, Y, y0)
    
    # Save as NPZ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pinn_three_body_dataset.npz"
    
    np.savez(
        filename,
        t=t,
        Y=Y,
        y0=y0,
        failed_indices=np.array(failed)
    )
    
    print(f"\nDataset saved to: {filename}")
    print(f"  t shape: {t.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  y0 shape: {y0.shape}")
    
    # Visualize samples
    visualize_samples(t, Y, num_samples=6, save_path="pinn_sample_trajectories.png")
    
    return filename


if __name__ == "__main__":
    main()
