"""
Generate animated GIFs of three-body trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


def load_dataset(filepath="pinn_three_body_dataset.npz"):
    """Load the dataset."""
    data = np.load(filepath)
    return data['t'], data['Y'], data['y0']


def create_single_trajectory_gif(t, trajectory, save_path="trajectory_animation.gif",
                                  trail_length=20, fps=15):
    """
    Create an animated GIF of a single trajectory with trailing paths.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['#e74c3c', '#3498db', '#2ecc71']
    labels = ['Body 1', 'Body 2', 'Body 3']

    # Extract all positions for axis limits
    all_x = []
    all_z = []
    for body in range(3):
        all_x.extend(trajectory[:, body * 3])
        all_z.extend(trajectory[:, body * 3 + 2])

    margin = 0.5
    xlim = (min(all_x) - margin, max(all_x) + margin)
    zlim = (min(all_z) - margin, max(all_z) + margin)

    # Make square
    max_range = max(xlim[1] - xlim[0], zlim[1] - zlim[0]) / 2
    x_center = (xlim[0] + xlim[1]) / 2
    z_center = (zlim[0] + zlim[1]) / 2

    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(z_center - max_range, z_center + max_range)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Z', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    # Initialize plot elements
    trails = []
    points = []

    for body in range(3):
        trail, = ax.plot([], [], color=colors[body], alpha=0.6, linewidth=2, label=labels[body])
        point, = ax.plot([], [], 'o', color=colors[body], markersize=12,
                        markeredgecolor='black', markeredgewidth=1.5)
        trails.append(trail)
        points.append(point)

    ax.legend(loc='upper right', fontsize=10)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    title = ax.set_title('Three-Body Problem', fontsize=14, fontweight='bold')

    def init():
        for trail, point in zip(trails, points):
            trail.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return trails + points + [time_text]

    def animate(frame):
        start = max(0, frame - trail_length)

        for body in range(3):
            x = trajectory[start:frame+1, body * 3]
            z = trajectory[start:frame+1, body * 3 + 2]
            trails[body].set_data(x, z)

            if frame < len(trajectory):
                points[body].set_data([trajectory[frame, body * 3]],
                                     [trajectory[frame, body * 3 + 2]])

        time_text.set_text(f't = {t[frame]:.2f}')
        return trails + points + [time_text]

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                        interval=1000//fps, blit=True)

    print(f"Saving {save_path}...")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def create_multi_trajectory_gif(t, Y, indices, save_path="multi_trajectory.gif",
                                 trail_length=15, fps=15):
    """
    Create an animated GIF showing multiple trajectories side by side.
    """
    n_traj = len(indices)
    fig, axes = plt.subplots(1, n_traj, figsize=(6*n_traj, 6))
    if n_traj == 1:
        axes = [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71']

    all_trails = []
    all_points = []
    time_texts = []

    for ax_idx, (ax, traj_idx) in enumerate(zip(axes, indices)):
        trajectory = Y[traj_idx]

        # Get limits
        all_x, all_z = [], []
        for body in range(3):
            all_x.extend(trajectory[:, body * 3])
            all_z.extend(trajectory[:, body * 3 + 2])

        margin = 0.5
        max_range = max(max(all_x) - min(all_x), max(all_z) - min(all_z)) / 2 + margin
        x_center = (min(all_x) + max(all_x)) / 2
        z_center = (min(all_z) + max(all_z)) / 2

        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(z_center - max_range, z_center + max_range)
        ax.set_aspect('equal')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Z', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        ax.set_title(f'Trajectory {traj_idx}', fontsize=12, fontweight='bold')

        trails = []
        points = []
        for body in range(3):
            trail, = ax.plot([], [], color=colors[body], alpha=0.6, linewidth=2)
            point, = ax.plot([], [], 'o', color=colors[body], markersize=10,
                            markeredgecolor='black', markeredgewidth=1)
            trails.append(trail)
            points.append(point)

        all_trails.append(trails)
        all_points.append(points)

        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        time_texts.append(time_text)

    plt.tight_layout()

    def init():
        elements = []
        for trails, points in zip(all_trails, all_points):
            for trail, point in zip(trails, points):
                trail.set_data([], [])
                point.set_data([], [])
                elements.extend([trail, point])
        for tt in time_texts:
            tt.set_text('')
            elements.append(tt)
        return elements

    def animate(frame):
        elements = []
        start = max(0, frame - trail_length)

        for ax_idx, traj_idx in enumerate(indices):
            trajectory = Y[traj_idx]

            for body in range(3):
                x = trajectory[start:frame+1, body * 3]
                z = trajectory[start:frame+1, body * 3 + 2]
                all_trails[ax_idx][body].set_data(x, z)

                if frame < len(trajectory):
                    all_points[ax_idx][body].set_data([trajectory[frame, body * 3]],
                                                      [trajectory[frame, body * 3 + 2]])

            elements.extend(all_trails[ax_idx])
            elements.extend(all_points[ax_idx])
            time_texts[ax_idx].set_text(f't = {t[frame]:.2f}')
            elements.append(time_texts[ax_idx])

        return elements

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                        interval=1000//fps, blit=True)

    print(f"Saving {save_path}...")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def create_chaos_comparison_gif(t, Y, y0, save_path="chaos_divergence.gif",
                                 trail_length=25, fps=15):
    """
    Create a GIF showing how similar initial conditions lead to diverging trajectories.
    This demonstrates the chaotic nature of the three-body problem.
    """
    # Find trajectories with similar initial conditions
    base_idx = 0
    distances = np.linalg.norm(y0 - y0[base_idx], axis=1)
    similar_indices = np.argsort(distances)[1:6]  # 5 most similar (excluding self)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Use a colormap for different trajectories
    cmap = plt.cm.viridis
    traj_colors = [cmap(i/5) for i in range(5)]

    # Get limits from all trajectories
    all_x, all_z = [], []
    for idx in similar_indices:
        trajectory = Y[idx]
        for body in range(3):
            all_x.extend(trajectory[:, body * 3])
            all_z.extend(trajectory[:, body * 3 + 2])

    margin = 1.0
    max_range = max(max(all_x) - min(all_x), max(all_z) - min(all_z)) / 2 + margin
    x_center = (min(all_x) + max(all_x)) / 2
    z_center = (min(all_z) + max(all_z)) / 2

    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(z_center - max_range, z_center + max_range)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Z', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.patch.set_facecolor('#1a1a2e')

    # We'll only track Body 1 to show divergence clearly
    trails = []
    points = []

    for i, idx in enumerate(similar_indices):
        trail, = ax.plot([], [], color=traj_colors[i], alpha=0.8, linewidth=2,
                        label=f'Traj {idx}')
        point, = ax.plot([], [], 'o', color=traj_colors[i], markersize=10,
                        markeredgecolor='white', markeredgewidth=1)
        trails.append(trail)
        points.append(point)

    ax.legend(loc='upper right', fontsize=10, facecolor='#2a2a4e',
              labelcolor='white', edgecolor='white')

    title = ax.set_title('Chaos in Three-Body Problem\n(Body 1 from similar initial conditions)',
                        fontsize=14, fontweight='bold', color='white')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', fontfamily='monospace', color='white',
                       bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9))

    def init():
        for trail, point in zip(trails, points):
            trail.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return trails + points + [time_text]

    def animate(frame):
        start = max(0, frame - trail_length)

        for i, idx in enumerate(similar_indices):
            trajectory = Y[idx]
            # Only Body 1
            x = trajectory[start:frame+1, 0]
            z = trajectory[start:frame+1, 2]
            trails[i].set_data(x, z)

            if frame < len(trajectory):
                points[i].set_data([trajectory[frame, 0]], [trajectory[frame, 2]])

        time_text.set_text(f't = {t[frame]:.2f}')
        return trails + points + [time_text]

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                        interval=1000//fps, blit=True)

    print(f"Saving {save_path}...")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading dataset...")
    t, Y, y0 = load_dataset()
    print(f"Loaded {len(Y)} trajectories")

    # Set random seed for reproducibility
    np.random.seed(42)

    print("\n--- Generating GIFs ---\n")

    # GIF 1: Single nice trajectory
    # Pick one with interesting dynamics (medium energy, has close encounters)
    interesting_idx = 42  # A nice example
    create_single_trajectory_gif(t, Y[interesting_idx],
                                  save_path="trajectory_single.gif",
                                  trail_length=25, fps=20)

    # GIF 2: Three trajectories side by side
    random_indices = np.random.choice(len(Y), 3, replace=False)
    create_multi_trajectory_gif(t, Y, random_indices.tolist(),
                                 save_path="trajectory_comparison.gif",
                                 trail_length=20, fps=20)

    # GIF 3: Chaos demonstration - similar ICs diverging
    create_chaos_comparison_gif(t, Y, y0,
                                 save_path="chaos_divergence.gif",
                                 trail_length=30, fps=20)

    print("\n" + "=" * 50)
    print("GIF GENERATION COMPLETE")
    print("=" * 50)
    print("\nGenerated files:")
    print("  1. trajectory_single.gif - Single trajectory animation")
    print("  2. trajectory_comparison.gif - 3 trajectories side by side")
    print("  3. chaos_divergence.gif - Chaos demonstration (diverging paths)")


if __name__ == "__main__":
    main()
