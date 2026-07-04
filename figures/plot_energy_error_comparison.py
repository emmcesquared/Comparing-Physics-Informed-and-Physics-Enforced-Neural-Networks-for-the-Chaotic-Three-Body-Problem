"""
Generate: Mean Absolute Energy Error over Time for All Four Models (± std across test set).

Models:
- DNN (baseline)
- PINN (baseline)
- LNN (bigger_model experiment)
- HNN-v2 (bigger_reduced experiment)

Output: energy_error_comparison.png in review3/
"""

import sys
import os
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent  # 90-s_trend/
DATASET = BASE / "dataset generation" / "pinn_three_body_dataset.npz"
OUTPUT = Path(__file__).resolve().parent / "energy_error_comparison.png"

MODEL_CONFIGS = {
    "DNN": {
        "model_dir": BASE / "models" / "dnn",
        "checkpoint": BASE / "models" / "dnn" / "checkpoints" / "best_model.pt",
        "color": "#2196F3",
    },
    "PINN": {
        "model_dir": BASE / "models" / "pinn",
        "checkpoint": BASE / "models" / "pinn" / "checkpoints" / "best_model.pt",
        "color": "#FF9800",
    },
    "LNN": {
        "model_dir": BASE / "models" / "lnn" / "experiments" / "20260115_165807_bigger_model" / "source_snapshot",
        "checkpoint": BASE / "models" / "lnn" / "experiments" / "20260115_165807_bigger_model" / "checkpoints" / "best_model.pt",
        "color": "#4CAF50",
    },
    "HNN": {
        "model_dir": BASE / "models" / "hnn" / "experiments" / "20260224_045832_bigger_reduced" / "source_snapshot",
        "checkpoint": BASE / "models" / "hnn" / "experiments" / "20260224_045832_bigger_reduced" / "checkpoints" / "best_model.pt",
        "color": "#E91E63",
    },
}


# ── helpers ────────────────────────────────────────────────────────────────

def load_module(module_name, file_path):
    """Dynamically import a module from an arbitrary path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_energy(positions, velocities, G=1.0, masses=None):
    """Compute total mechanical energy for three-body system. (T,6) -> (T,)"""
    if masses is None:
        masses = np.array([1.0, 1.0, 1.0])

    N = len(positions)
    energies = np.zeros(N)

    for i in range(N):
        pos = positions[i]
        vel = velocities[i]

        p = np.array([[pos[0], pos[1]],
                      [pos[2], pos[3]],
                      [pos[4], pos[5]]])
        v = np.array([[vel[0], vel[1]],
                      [vel[2], vel[3]],
                      [vel[4], vel[5]]])

        KE = 0.5 * np.sum(masses[:, np.newaxis] * v**2)
        PE = 0.0
        for j in range(3):
            for k in range(j + 1, 3):
                r_jk = np.linalg.norm(p[j] - p[k])
                PE -= G * masses[j] * masses[k] / max(r_jk, 1e-10)
        energies[i] = KE + PE

    return energies


def predict_trajectory(model, ic, t, norm_stats, device):
    """Standard prediction shared by all four architectures."""
    model.eval()
    T = len(t)

    inputs = np.zeros((T, 3), dtype=np.float32)
    inputs[:, 0] = t
    inputs[:, 1] = ic[0]
    inputs[:, 2] = ic[1]

    if norm_stats is not None:
        input_mean = np.array(norm_stats['input_mean'])
        input_std = np.array(norm_stats['input_std'])
        inputs = (inputs - input_mean) / input_std

    with torch.no_grad():
        inputs_t = torch.from_numpy(inputs).to(device)
        outputs = model(inputs_t).cpu().numpy()

    if norm_stats is not None:
        target_mean = np.array(norm_stats['target_mean'])
        target_std = np.array(norm_stats['target_std'])
        outputs = outputs * target_std + target_mean

    x1, z1 = outputs[:, 0], outputs[:, 1]
    x2, z2 = outputs[:, 2], outputs[:, 3]
    x3 = -(x1 + x2)
    z3 = -(z1 + z2)

    return np.column_stack([x1, z1, x2, z2, x3, z3])


def load_model_generic(name, cfg, device):
    """Load any of the four model types."""
    model_dir = cfg["model_dir"]
    ckpt_path = cfg["checkpoint"]
    config_path = ckpt_path.parent / "config.json"

    # Load model class from model.py in the model_dir
    model_mod = load_module(f"{name}_model", model_dir / "model.py")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    if name == "DNN":
        model = model_mod.ThreeBodyDNN(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 10),
            dropout=0.0,
        )
    elif name == "PINN":
        model = model_mod.ThreeBodyPINN(
            hidden_dim=config.get('hidden_dim', 128),
            num_blocks=config.get('num_blocks', 5),
        )
    elif name == "LNN":
        model = model_mod.ThreeBodyLNN(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 10),
            activation=config.get('activation', 'tanh'),
            lagrangian_hidden=config.get('lagrangian_hidden', 64),
            lagrangian_layers=config.get('lagrangian_layers', 3),
            use_separable_lagrangian=config.get('use_separable_lagrangian', True),
        )
    elif name == "HNN":
        model = model_mod.ThreeBodyHNN(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 10),
            activation=config.get('activation', 'tanh'),
            hamiltonian_hidden=config.get('hamiltonian_hidden', 64),
            hamiltonian_layers=config.get('hamiltonian_layers', 3),
            use_separable_hamiltonian=config.get('use_separable_hamiltonian', True),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    norm_stats = checkpoint.get('norm_stats', None)
    return model, norm_stats


# ── dataset loader (minimal, avoids import conflicts) ─────────────────────

def load_test_trajectories(data_path):
    """Load test split trajectories from the shared dataset.

    Dataset format:
        t: (129,)
        Y: (1000, 129, 18) — state = [x1,y1,z1,x2,y2,z2,x3,y3,z3,
                                        vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]
        y0: (1000, 18) — initial conditions
    Split: seed=42, random permutation, 80/10/10 train/val/test
    """
    data = np.load(str(data_path))
    t = data['t']        # (129,)
    Y = data['Y']        # (1000, 129, 18)
    y0 = data['y0']      # (1000, 18)

    n_traj = len(Y)
    np.random.seed(42)
    indices = np.random.permutation(n_traj)

    n_train = int(0.8 * n_traj)
    n_val = int(0.1 * n_traj)
    test_indices = indices[n_train + n_val:]

    # Extract x,z positions and velocities for test trajectories
    n_test = len(test_indices)
    T = len(t)

    positions = np.zeros((n_test, T, 6), dtype=np.float32)
    velocities = np.zeros((n_test, T, 6), dtype=np.float32)
    ics = np.zeros((n_test, 2), dtype=np.float32)

    for i, idx in enumerate(test_indices):
        traj = Y[idx]  # (129, 18)
        # Positions: x1=0, z1=2, x2=3, z2=5, x3=6, z3=8
        positions[i, :, 0] = traj[:, 0]   # x1
        positions[i, :, 1] = traj[:, 2]   # z1
        positions[i, :, 2] = traj[:, 3]   # x2
        positions[i, :, 3] = traj[:, 5]   # z2
        positions[i, :, 4] = traj[:, 6]   # x3
        positions[i, :, 5] = traj[:, 8]   # z3

        # Velocities: vx1=9, vz1=11, vx2=12, vz2=14, vx3=15, vz3=17
        velocities[i, :, 0] = traj[:, 9]   # vx1
        velocities[i, :, 1] = traj[:, 11]  # vz1
        velocities[i, :, 2] = traj[:, 12]  # vx2
        velocities[i, :, 3] = traj[:, 14]  # vz2
        velocities[i, :, 4] = traj[:, 15]  # vx3
        velocities[i, :, 5] = traj[:, 17]  # vz3

        # IC: [x2_0, z2_0]
        ics[i, 0] = y0[idx, 3]
        ics[i, 1] = y0[idx, 5]

    return t, positions, velocities, ics


# ── main ──────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test data
    print("Loading test data...")
    t, test_pos, test_vel, test_ics = load_test_trajectories(DATASET)
    n_test = len(test_pos)
    T = len(t)
    print(f"  {n_test} test trajectories, {T} timesteps")

    # For each model: compute per-timestep energy error across all test trajectories
    results = {}  # name -> (T,) arrays: mean, std

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name}...")
        print(f"{'='*50}")

        model, norm_stats = load_model_generic(name, cfg, device)

        # Collect per-timestep energy errors: (n_test, T)
        energy_errs = np.zeros((n_test, T))

        for i in tqdm(range(n_test), desc=f"  {name}"):
            ic = test_ics[i]
            true_p = test_pos[i]   # (T, 6)
            true_v = test_vel[i]   # (T, 6)

            pred_p = predict_trajectory(model, ic, t, norm_stats, device)

            E_true = compute_energy(true_p, true_v)
            E_pred = compute_energy(pred_p, true_v)  # use true velocities (position-only models)
            energy_errs[i] = np.abs(E_pred - E_true)

        mean_err = energy_errs.mean(axis=0)
        std_err = energy_errs.std(axis=0)
        results[name] = (mean_err, std_err)
        print(f"  Mean energy error (avg over time): {mean_err.mean():.4f}")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results for fast re-plotting
    save_data = {name: {'mean': results[name][0], 'std': results[name][1]}
                 for name in results}
    save_data['t'] = t
    np.savez(str(OUTPUT.with_suffix('.npz')), **{
        f'{name}_mean': results[name][0] for name in results
    }, **{
        f'{name}_std': results[name][1] for name in results
    }, t=t)
    print(f"Saved raw data to {OUTPUT.with_suffix('.npz')}")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nGenerating plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: linear scale with clipped y-axis (median-based cap)
    ax = axes[0]
    for name, cfg in MODEL_CONFIGS.items():
        mean_err, std_err = results[name]
        color = cfg["color"]
        ax.plot(t, mean_err, color=color, linewidth=2, label=name)
        ax.fill_between(t,
                        np.maximum(mean_err - std_err, 0),
                        mean_err + std_err,
                        color=color, alpha=0.15)

    # Cap y-axis at a reasonable value to avoid spike domination
    all_means = np.array([results[n][0] for n in results])
    y_cap = np.percentile(all_means, 95) * 3
    ax.set_ylim(0, max(y_cap, 10))
    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Mean Absolute Energy Error', fontsize=13)
    ax.set_title('Energy Error Over Time (linear scale)', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    # Right panel: log scale (shows full dynamic range)
    ax = axes[1]
    for name, cfg in MODEL_CONFIGS.items():
        mean_err, std_err = results[name]
        color = cfg["color"]
        ax.semilogy(t, mean_err, color=color, linewidth=2, label=name)
        ax.fill_between(t,
                        np.maximum(mean_err - std_err, 1e-6),
                        mean_err + std_err,
                        color=color, alpha=0.15)

    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Mean Absolute Energy Error (log)', fontsize=13)
    ax.set_title('Energy Error Over Time (log scale)', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(t[0], t[-1])

    plt.suptitle('Mean Absolute Energy Error ± std across test set', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(str(OUTPUT), dpi=200, bbox_inches='tight')
    print(f"\nSaved to {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
