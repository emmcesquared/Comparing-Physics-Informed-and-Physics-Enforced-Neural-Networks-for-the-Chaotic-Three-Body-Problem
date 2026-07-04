"""
HNN Experiment Launcher with Timestamped Folders.

Each experiment run creates a new timestamped folder containing:
- Model checkpoints (best_model.pt, final_model.pt)
- Training config (config.json)
- Training history (history.npz)
- Evaluation results (metrics.json, plots)
- Model architecture snapshot

This allows systematic exploration of HNN architectures for the three-body problem.

Usage:
    python run_experiment.py --hidden-dim 128 --num-layers 4 --note "baseline"
    python run_experiment.py --hidden-dim 256 --num-layers 6 --note "larger_model"
    python run_experiment.py --list
    python run_experiment.py --eval-only 20260213_153000_baseline
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


def create_experiment_folder(base_dir, note=""):
    """
    Create a timestamped experiment folder.

    Format: YYYYMMDD_HHMMSS_note/

    Returns:
        experiment_dir: Path to the new experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if note:
        note_clean = note.replace(" ", "_").replace("/", "-")[:50]
        folder_name = f"{timestamp}_{note_clean}"
    else:
        folder_name = timestamp

    experiment_dir = Path(base_dir) / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "eval_results").mkdir(exist_ok=True)

    return experiment_dir


def save_experiment_metadata(experiment_dir, args, extra_info=None):
    """Save experiment metadata for reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "args": vars(args),
        "note": args.note if hasattr(args, 'note') else "",
    }

    if extra_info:
        metadata.update(extra_info)

    with open(experiment_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def copy_source_files(experiment_dir, source_dir):
    """Copy source files to experiment folder for reproducibility."""
    source_snapshot_dir = experiment_dir / "source_snapshot"
    source_snapshot_dir.mkdir(exist_ok=True)

    files_to_copy = ["model.py", "dataset.py", "train.py", "evaluate.py", "metrics.py"]

    for fname in files_to_copy:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, source_snapshot_dir / fname)


def list_experiments(base_dir):
    """List all experiment folders with their metadata."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print("No experiments directory found.")
        return

    experiments = []
    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir():
            metadata_file = folder / "experiment_metadata.json"
            config_file = folder / "checkpoints" / "config.json"
            metrics_file = folder / "eval_results" / "metrics.json"

            info = {"name": folder.name, "path": str(folder)}

            if metadata_file.exists():
                with open(metadata_file) as f:
                    meta = json.load(f)
                    info["note"] = meta.get("note", "")
                    info["timestamp"] = meta.get("timestamp", "")

            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    info["hidden_dim"] = config.get("hidden_dim", "?")
                    info["num_layers"] = config.get("num_layers", "?")
                    info["n_params"] = config.get("n_params", "?")
                    info["hamiltonian_hidden"] = config.get("hamiltonian_hidden", "?")

            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    info["mae_mean"] = metrics.get("mae_mean", "?")
                    info["hamiltonian_residual_mean"] = metrics.get("hamiltonian_residual_mean", "?")

            experiments.append(info)

    print("\n" + "="*110)
    print("HNN EXPERIMENTS")
    print("="*110)
    print(f"{'Name':<30} {'Note':<20} {'Hidden':<8} {'Layers':<8} {'H-Net':<8} {'MAE':<12} {'H Resid':<12}")
    print("-"*110)

    for exp in experiments:
        mae = f"{exp.get('mae_mean', '?'):.4f}" if isinstance(exp.get('mae_mean'), float) else "?"
        h_res = f"{exp.get('hamiltonian_residual_mean', '?'):.4f}" if isinstance(exp.get('hamiltonian_residual_mean'), float) else "?"
        print(f"{exp['name']:<30} {exp.get('note', ''):<20} {exp.get('hidden_dim', '?'):<8} {exp.get('num_layers', '?'):<8} {exp.get('hamiltonian_hidden', '?'):<8} {mae:<12} {h_res:<12}")

    print("="*110 + "\n")


def run_training(experiment_dir, args):
    """Run training with the given configuration."""
    from train import train

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_dir.name}")
    print(f"{'='*70}\n")

    model, history = train(
        data_path=args.data,
        output_dir=experiment_dir / "checkpoints",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        hamiltonian_hidden=args.hamiltonian_hidden,
        hamiltonian_layers=args.hamiltonian_layers,
        lambda_physics=args.lambda_physics,
        lambda_hamilton=args.lambda_hamilton,
        lambda_ham_sup=args.lambda_ham_sup,
        use_separable_hamiltonian=not args.no_separable_hamiltonian,
        n_collocation=args.n_collocation,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        patience=args.patience,
        device=args.device
    )

    return model, history


def run_evaluation(experiment_dir, args):
    """Run evaluation on the trained model."""
    from evaluate import main as evaluate_main
    import sys

    checkpoint_path = experiment_dir / "checkpoints" / "best_model.pt"
    output_dir = experiment_dir / "eval_results"

    eval_args = [
        "--checkpoint", str(checkpoint_path),
        "--data", args.data,
        "--output", str(output_dir),
        "--device", args.device
    ]

    old_argv = sys.argv
    sys.argv = ["evaluate.py"] + eval_args

    try:
        evaluate_main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description='HNN Experiment Launcher')

    # Experiment management
    parser.add_argument('--list', action='store_true',
                        help='List all experiments')
    parser.add_argument('--note', type=str, default='',
                        help='Experiment note/description')
    parser.add_argument('--experiments-dir', type=str, default='./experiments',
                        help='Base directory for experiments')

    # Data
    parser.add_argument('--data', type=str,
                        default='../../dataset generation/pinn_three_body_dataset.npz',
                        help='Path to dataset')

    # Model architecture (position network)
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension for position network')
    parser.add_argument('--num-layers', type=int, default=10,
                        help='Number of hidden layers for position network')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['softplus', 'tanh', 'relu', 'elu'],
                        help='Activation function')

    # Hamiltonian network architecture
    parser.add_argument('--hamiltonian-hidden', type=int, default=64,
                        help='Hidden dimension for Hamiltonian network')
    parser.add_argument('--hamiltonian-layers', type=int, default=3,
                        help='Number of layers for Hamiltonian network')
    parser.add_argument('--no-separable-hamiltonian', action='store_true',
                        help='Disable separable H=T+V structure (use unconstrained Hamiltonian)')

    # Training
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda-physics', type=float, default=0.1,
                        help='Weight for physics (ODE) loss')
    parser.add_argument('--lambda-hamilton', type=float, default=0.1,
                        help='Weight for Hamilton equations loss')
    parser.add_argument('--lambda-ham-sup', type=float, default=1.0,
                        help='Weight for Hamiltonian supervision loss (H_learned vs H_true)')
    parser.add_argument('--n-collocation', type=int, default=10000,
                        help='Number of collocation points')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')

    # Workflow
    parser.add_argument('--train-only', action='store_true',
                        help='Only run training, skip evaluation')
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Only run evaluation on existing experiment folder')

    args = parser.parse_args()

    base_dir = Path(args.experiments_dir)

    # Handle --list
    if args.list:
        list_experiments(base_dir)
        return

    # Handle --eval-only
    if args.eval_only:
        experiment_dir = base_dir / args.eval_only
        if not experiment_dir.exists():
            print(f"Error: Experiment folder not found: {experiment_dir}")
            return
        print(f"Running evaluation on: {experiment_dir.name}")
        run_evaluation(experiment_dir, args)
        return

    # Create new experiment
    experiment_dir = create_experiment_folder(base_dir, args.note)
    print(f"\nCreated experiment folder: {experiment_dir}")

    # Save metadata and source snapshot
    save_experiment_metadata(experiment_dir, args)
    copy_source_files(experiment_dir, Path(__file__).parent)

    # Run training
    model, history = run_training(experiment_dir, args)

    # Run evaluation (unless --train-only)
    if not args.train_only:
        print("\n" + "="*70)
        print("Running Evaluation...")
        print("="*70 + "\n")
        run_evaluation(experiment_dir, args)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {experiment_dir.name}")
    print(f"{'='*70}")
    print(f"Results saved to: {experiment_dir}")
    print(f"\nTo view all experiments: python run_experiment.py --list")


if __name__ == '__main__':
    main()
