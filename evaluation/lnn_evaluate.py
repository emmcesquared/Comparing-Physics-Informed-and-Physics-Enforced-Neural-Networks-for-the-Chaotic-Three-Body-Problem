"""
Evaluation script for Three-Body LNN.

SAME EVALUATION AS DNN/PINN - direct position prediction, same metrics.
Plus LNN-specific metrics (E-L residual).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ThreeBodyLNN
from dataset import LNNTrajectoryDataset
from metrics import (
    evaluate_trajectory_with_ode,
    aggregate_metrics,
    print_metrics,
    compute_energy,
    # LNN-specific (Tier B)
    evaluate_trajectory_lnn,
    aggregate_metrics_lnn,
    print_metrics_lnn
)


def load_model(checkpoint_path, device='cuda'):
    """Load trained LNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model = ThreeBodyLNN(
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 10),
            activation=config.get('activation', 'tanh'),
            lagrangian_hidden=config.get('lagrangian_hidden', 64),
            lagrangian_layers=config.get('lagrangian_layers', 3),
            use_separable_lagrangian=config.get('use_separable_lagrangian', True)
        )
    else:
        model = ThreeBodyLNN()
        config = {}

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    norm_stats = checkpoint.get('norm_stats', None)

    return model, norm_stats, config


def predict_trajectory(model, ic, t, norm_stats, device='cuda'):
    """
    Predict full trajectory given initial conditions.
    SAME AS DNN - direct query at each time step.

    Args:
        model: Trained LNN model
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


def evaluate_model(model, data_path, norm_stats, device='cuda', split='test',
                   include_lnn_metrics=True):
    """
    Evaluate model on all test trajectories.

    Includes both Tier A (common) and Tier B (LNN-specific) metrics.

    Returns:
        all_metrics: list of per-trajectory metrics
        aggregated: aggregated statistics
    """
    dataset = LNNTrajectoryDataset(data_path, split=split)

    all_metrics = []

    print(f"Evaluating on {len(dataset)} trajectories...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        ic = sample['ic'].numpy()
        t = sample['t'].numpy()
        true_pos = sample['positions'].numpy()
        true_vel = sample['velocities'].numpy()

        # Predict (same as DNN)
        pred_pos = predict_trajectory(model, ic, t, norm_stats, device)

        # Compute metrics
        if include_lnn_metrics:
            # Tier A + Tier B (LNN-specific: E-L Residual, Lagrangian Structure)
            metrics = evaluate_trajectory_lnn(
                model, pred_pos, true_pos, true_vel, t, ic, norm_stats, device
            )
        else:
            # Tier A only (same as DNN)
            metrics = evaluate_trajectory_with_ode(pred_pos, true_pos, true_vel, t)

        metrics['traj_idx'] = sample['traj_idx']
        all_metrics.append(metrics)

    # Aggregate with LNN-specific metrics
    if include_lnn_metrics:
        aggregated = aggregate_metrics_lnn(all_metrics)
    else:
        aggregated = aggregate_metrics(all_metrics)

    return all_metrics, aggregated


def plot_training_history(history_path, save_path=None):
    """Plot training history."""
    history = np.load(history_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    ax = axes[0, 0]
    ax.semilogy(history['train_loss'], label='Train', alpha=0.8)
    ax.semilogy(history['val_loss'], label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Component losses
    ax = axes[0, 1]
    if 'data_loss' in history:
        ax.semilogy(history['data_loss'], label='Data', alpha=0.8)
    if 'physics_loss' in history:
        ax.semilogy(history['physics_loss'], label='Physics (ODE)', alpha=0.8)
    if 'el_loss' in history:
        ax.semilogy(history['el_loss'], label='E-L Residual', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.semilogy(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    # Validation loss (linear)
    ax = axes[1, 1]
    ax.plot(history['val_loss'], 'g-', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss (Linear Scale)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_predictions(model, data_path, norm_stats, device='cuda',
                            n_samples=3, save_path=None):
    """Plot sample trajectory predictions vs ground truth (side-by-side)."""
    dataset = LNNTrajectoryDataset(data_path, split='test')

    # 3 samples x 2 columns (True, Predicted)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 5*n_samples))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green for 3 bodies

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for row_idx, data_idx in enumerate(indices):
        sample = dataset[data_idx]

        ic = sample['ic'].numpy()
        t = sample['t'].numpy()
        true_pos = sample['positions'].numpy()
        pred_pos = predict_trajectory(model, ic, t, norm_stats, device)

        # Left column: Ground Truth
        ax_true = axes[row_idx, 0]
        for body in range(3):
            ax_true.plot(true_pos[:, body*2], true_pos[:, body*2+1],
                        color=colors[body], linewidth=2, label=f'Body {body+1}')
            ax_true.scatter(true_pos[0, body*2], true_pos[0, body*2+1],
                           color=colors[body], marker='o', s=80, edgecolor='black', zorder=5)
        ax_true.set_xlabel('X')
        ax_true.set_ylabel('Z')
        ax_true.set_title(f'Trajectory {sample["traj_idx"]} - Ground Truth')
        ax_true.grid(True, alpha=0.3)
        ax_true.set_aspect('equal')
        if row_idx == 0:
            ax_true.legend(fontsize=8, loc='upper right')

        # Right column: Model Prediction
        ax_pred = axes[row_idx, 1]
        for body in range(3):
            ax_pred.plot(pred_pos[:, body*2], pred_pos[:, body*2+1],
                        color=colors[body], linewidth=2, label=f'Body {body+1}')
            ax_pred.scatter(pred_pos[0, body*2], pred_pos[0, body*2+1],
                           color=colors[body], marker='o', s=80, edgecolor='black', zorder=5)
        ax_pred.set_xlabel('X')
        ax_pred.set_ylabel('Z')
        ax_pred.set_title(f'Trajectory {sample["traj_idx"]} - LNN Prediction')
        ax_pred.grid(True, alpha=0.3)
        ax_pred.set_aspect('equal')
        
        # Match axis limits between true and pred
        xlim = (min(ax_true.get_xlim()[0], ax_pred.get_xlim()[0]),
                max(ax_true.get_xlim()[1], ax_pred.get_xlim()[1]))
        ylim = (min(ax_true.get_ylim()[0], ax_pred.get_ylim()[0]),
                max(ax_true.get_ylim()[1], ax_pred.get_ylim()[1]))
        ax_true.set_xlim(xlim)
        ax_pred.set_xlim(xlim)
        ax_true.set_ylim(ylim)
        ax_pred.set_ylim(ylim)

    plt.suptitle('LNN: Ground Truth vs Prediction (●=start)', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_analysis(all_metrics, t, save_path=None):
    """Plot error analysis across trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

    # 3. Distribution of MAE
    ax = axes[1, 0]
    maes = [m['mae'] for m in all_metrics]
    ax.hist(maes, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(maes), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(maes):.4f}')
    ax.set_xlabel('MAE')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of MAE across Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distribution of long-horizon error
    ax = axes[1, 1]
    lh_errs = [m['long_horizon_err'] for m in all_metrics]
    ax.hist(lh_errs, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(lh_errs), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(lh_errs):.4f}')
    ax.set_xlabel('Long-Horizon Relative Error')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Long-Horizon Error (t=10)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(aggregated, all_metrics, output_dir):
    """Save evaluation results to files."""
    output_dir = Path(output_dir)

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
        'energy_drift': [m['energy_drift'] for m in all_metrics]
    }
    if 'ode_residual' in all_metrics[0]:
        per_traj['ode_residual'] = [m['ode_residual'] for m in all_metrics]
    
    # LNN-specific Tier B metrics
    if 'el_residual' in all_metrics[0]:
        per_traj['el_residual'] = [m['el_residual'] for m in all_metrics]
    if 'lagrangian_structure_score' in all_metrics[0]:
        per_traj['lagrangian_structure_score'] = [m['lagrangian_structure_score'] for m in all_metrics]

    np.savez(output_dir / 'per_trajectory_metrics.npz', **per_traj)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Three-Body LNN')
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
    model, norm_stats, config = load_model(args.checkpoint, device)
    print(f"Model: {config.get('hidden_dim', '?')} hidden, {config.get('num_layers', '?')} layers")

    # Evaluate
    all_metrics, aggregated = evaluate_model(
        model, args.data, norm_stats, device, split='test'
    )

    # Print results (Tier A + Tier B LNN-specific)
    print_metrics_lnn(aggregated, model_name="LNN")

    # Load time array
    data = np.load(args.data)
    t = data['t']

    # Generate plots
    print("\nGenerating plots...")

    # Training history
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
