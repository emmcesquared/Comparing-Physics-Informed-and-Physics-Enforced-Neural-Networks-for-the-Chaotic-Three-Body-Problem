"""
Evaluation script for Three-Body PINN.

Computes:
- All Tier A metrics (same as DNN)
- PINN-specific: ODE Residual

Generates visualizations and saves results.
"""

import argparse
import json
from pathlib import Path
import sys
import os
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ThreeBodyPINN
from metrics import (
    evaluate_trajectory_pinn,
    aggregate_metrics_pinn,
    print_metrics_pinn
)

# Load DNN dataset module explicitly to avoid name collision
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dnn_dataset_path = os.path.join(os.path.dirname(_current_dir), 'dnn', 'dataset.py')
_spec = importlib.util.spec_from_file_location("dnn_dataset", _dnn_dataset_path)
_dnn_dataset_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dnn_dataset_module)

ThreeBodyTrajectoryDataset = _dnn_dataset_module.ThreeBodyTrajectoryDataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config to get model architecture
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model = ThreeBodyPINN(
            hidden_dim=config.get('hidden_dim', 128),
            num_blocks=config.get('num_blocks', 5)
        )
    else:
        model = ThreeBodyPINN()

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    norm_stats = checkpoint.get('norm_stats', None)

    return model, norm_stats


def predict_trajectory(model, ic, t, norm_stats, device='cuda'):
    """
    Predict full trajectory given initial conditions.

    Args:
        model: Trained PINN model
        ic: Initial conditions [x2_0, z2_0]
        t: Time array (T,)
        norm_stats: Normalization statistics
        device: torch device

    Returns:
        pred_positions: (T, 6) predicted positions [x1,z1,x2,z2,x3,z3]
    """
    model.eval()
    T = len(t)

    # Build input: [t, x2_0, z2_0] for each time step
    inputs = np.zeros((T, 3), dtype=np.float32)
    inputs[:, 0] = t
    inputs[:, 1] = ic[0]  # x2_0
    inputs[:, 2] = ic[1]  # z2_0

    # Normalize
    if norm_stats is not None:
        input_mean = np.array(norm_stats['input_mean'])
        input_std = np.array(norm_stats['input_std'])
        inputs = (inputs - input_mean) / input_std

    # Predict
    with torch.no_grad():
        inputs_t = torch.from_numpy(inputs).to(device)
        outputs = model(inputs_t).cpu().numpy()

    # Denormalize
    if norm_stats is not None:
        target_mean = np.array(norm_stats['target_mean'])
        target_std = np.array(norm_stats['target_std'])
        outputs = outputs * target_std + target_mean

    # outputs: [x1, z1, x2, z2]
    # Compute body 3 from COM constraint
    x1, z1 = outputs[:, 0], outputs[:, 1]
    x2, z2 = outputs[:, 2], outputs[:, 3]
    x3 = -(x1 + x2)
    z3 = -(z1 + z2)

    pred_positions = np.column_stack([x1, z1, x2, z2, x3, z3])
    return pred_positions


def evaluate_model(model, data_path, norm_stats, device='cuda', split='test'):
    """
    Evaluate model on all test trajectories.

    Returns:
        all_metrics: list of per-trajectory metrics
        aggregated: aggregated statistics
    """
    dataset = ThreeBodyTrajectoryDataset(data_path, split=split)

    # Get time array
    data = np.load(data_path)
    t = data['t']

    all_metrics = []

    print(f"Evaluating on {len(dataset)} trajectories...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        ic = sample['ic'].numpy()
        true_pos = sample['positions'].numpy()
        true_vel = sample['velocities'].numpy()

        # Predict
        pred_pos = predict_trajectory(model, ic, t, norm_stats, device)

        # Compute metrics (Tier A + PINN)
        metrics = evaluate_trajectory_pinn(pred_pos, true_pos, true_vel, t)
        metrics['traj_idx'] = sample['traj_idx']
        all_metrics.append(metrics)

    aggregated = aggregate_metrics_pinn(all_metrics)
    return all_metrics, aggregated


def plot_training_history(history_path, save_path=None):
    """Plot training history with data and physics loss."""
    history = np.load(history_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    ax = axes[0, 0]
    ax.semilogy(history['train_loss'], label='Train', alpha=0.8)
    ax.semilogy(history['val_loss'], label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Data + Physics)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Data vs Physics loss
    ax = axes[0, 1]
    ax.semilogy(history['data_loss'], label='Data Loss', alpha=0.8)
    ax.semilogy(history['physics_loss'], label='Physics Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Data vs Physics Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.semilogy(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    # Loss ratio
    ax = axes[1, 1]
    ratio = np.array(history['physics_loss']) / (np.array(history['data_loss']) + 1e-10)
    ax.plot(ratio, alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1:1 ratio')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics/Data Loss Ratio')
    ax.set_title('Loss Component Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_predictions(model, data_path, norm_stats, device='cuda',
                            n_samples=6, save_path=None):
    """Plot sample trajectory predictions vs ground truth."""
    dataset = ThreeBodyTrajectoryDataset(data_path, split='test')
    data = np.load(data_path)
    t = data['t']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors_true = ['#e74c3c', '#3498db', '#2ecc71']
    colors_pred = ['#c0392b', '#2980b9', '#27ae60']

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for ax_idx, data_idx in enumerate(indices):
        ax = axes[ax_idx]
        sample = dataset[data_idx]

        ic = sample['ic'].numpy()
        true_pos = sample['positions'].numpy()

        pred_pos = predict_trajectory(model, ic, t, norm_stats, device)

        # Plot each body
        for body in range(3):
            # True
            ax.plot(true_pos[:, body*2], true_pos[:, body*2+1],
                   color=colors_true[body], alpha=0.7, linewidth=2,
                   label=f'Body {body+1} (true)')
            # Predicted
            ax.plot(pred_pos[:, body*2], pred_pos[:, body*2+1],
                   color=colors_pred[body], linestyle='--', alpha=0.9, linewidth=1.5,
                   label=f'Body {body+1} (pred)')

            # Start markers
            ax.scatter(true_pos[0, body*2], true_pos[0, body*2+1],
                      color=colors_true[body], marker='o', s=60, edgecolor='black', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Trajectory {sample["traj_idx"]}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if ax_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('PINN Predictions vs Ground Truth\n(solid=true, dashed=predicted)', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_analysis(all_metrics, t, save_path=None):
    """Plot error analysis including ODE residual."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Relative error over time
    ax = axes[0, 0]
    rel_errs = np.array([m['rel_err_per_step'] for m in all_metrics])
    mean_err = rel_errs.mean(axis=0)
    std_err = rel_errs.std(axis=0)

    ax.plot(t, mean_err, 'b-', linewidth=2, label='Mean')
    ax.fill_between(t, mean_err - std_err, mean_err + std_err, alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Energy error over time
    ax = axes[0, 1]
    energy_errs = np.array([m['energy_err_per_step'] for m in all_metrics])
    mean_e_err = energy_errs.mean(axis=0)
    std_e_err = energy_errs.std(axis=0)

    ax.plot(t, mean_e_err, 'r-', linewidth=2, label='Mean')
    ax.fill_between(t, mean_e_err - std_e_err, mean_e_err + std_e_err,
                   alpha=0.3, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute Energy Error')
    ax.set_title('Energy Error over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ODE Residual over time
    ax = axes[0, 2]
    ode_res = np.array([m['ode_res_per_step'] for m in all_metrics])
    mean_ode = ode_res.mean(axis=0)
    std_ode = ode_res.std(axis=0)

    ax.plot(t, mean_ode, 'g-', linewidth=2, label='Mean')
    ax.fill_between(t, mean_ode - std_ode, mean_ode + std_ode,
                   alpha=0.3, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('ODE Residual')
    ax.set_title('ODE Residual over Time (PINN-specific)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distribution of MAE
    ax = axes[1, 0]
    maes = [m['mae'] for m in all_metrics]
    ax.hist(maes, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(maes), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(maes):.4f}')
    ax.set_xlabel('MAE')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Distribution of long-horizon error
    ax = axes[1, 1]
    lh_errs = [m['long_horizon_err'] for m in all_metrics]
    ax.hist(lh_errs, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(lh_errs), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(lh_errs):.4f}')
    ax.set_xlabel('Long-Horizon Relative Error')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Long-Horizon Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Distribution of ODE residual
    ax = axes[1, 2]
    ode_vals = [m['ode_residual'] for m in all_metrics]
    ax.hist(ode_vals, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(ode_vals), color='purple', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(ode_vals):.4f}')
    ax.set_xlabel('ODE Residual')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of ODE Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(aggregated, all_metrics, output_dir):
    """Save evaluation results to files."""
    output_dir = Path(output_dir)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save aggregated metrics as JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(convert_to_native(aggregated), f, indent=2)

    # Save per-trajectory metrics
    per_traj = {
        'traj_idx': [m['traj_idx'] for m in all_metrics],
        'mae': [m['mae'] for m in all_metrics],
        'rmse': [m['rmse'] for m in all_metrics],
        'rollout_err_avg': [m['rollout_err_avg'] for m in all_metrics],
        'long_horizon_err': [m['long_horizon_err'] for m in all_metrics],
        'energy_drift': [m['energy_drift'] for m in all_metrics],
        'ode_residual': [m['ode_residual'] for m in all_metrics]
    }
    np.savez(output_dir / 'per_trajectory_metrics.npz', **per_traj)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Three-Body PINN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                        default='../../dataset generation/pinn_three_body_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='./eval_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, norm_stats = load_model(args.checkpoint, device)

    # Evaluate
    all_metrics, aggregated = evaluate_model(
        model, args.data, norm_stats, device, split='test'
    )

    # Print results
    print_metrics_pinn(aggregated, model_name="PINN")

    # Load time array for plotting
    data = np.load(args.data)
    t = data['t']

    # Generate plots
    print("\nGenerating plots...")

    # Training history (if available)
    history_path = Path(args.checkpoint).parent / 'history.npz'
    if history_path.exists():
        plot_training_history(history_path, output_dir / 'training_history.png')
        print("  - training_history.png")

    plot_sample_predictions(model, args.data, norm_stats, device,
                            save_path=output_dir / 'sample_predictions.png')
    print("  - sample_predictions.png")

    plot_error_analysis(all_metrics, t, save_path=output_dir / 'error_analysis.png')
    print("  - error_analysis.png")

    # Save results
    save_results(aggregated, all_metrics, output_dir)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
